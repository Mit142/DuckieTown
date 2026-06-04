#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_follow.py
==============
Duckiebot lane-following controller (ROS Noetic).

WHAT IT DOES, in one sentence:
  Look at the camera -> find the yellow (left) and white (right) lane lines ->
  work out how far off-centre the bot is -> turn that error into a steering
  command -> publish wheel commands so the bot stays in the lane.

=============================================================================
 IF IT IS "NOT DRIVING AT ALL", it is almost always ONE of these three.
 Each is marked in the code below with a  >>> NOT DRIVING #n <<<  comment.

   #1  Commands are published but IGNORED downstream.
       -> the car-command switch is in manual/joystick mode, or the cmd topic
          is wrong for your distro. (Test A: manually `rostopic pub` a command.)

   #2  The node publishes v=0 because it thinks the LANE IS LOST.
       -> detection found no lines for > lost_timeout seconds, so it stops on
          purpose. Usually HSV thresholds vs lighting. (Test B: `rostopic echo`
          the cmd topic -> you'll see v: 0.0 constantly.)

   #3  The node never publishes anything because NO CAMERA FRAMES arrive.
       -> self.latest stays None, the loop never reaches publish_cmd().
          (Check VEHICLE_NAME, and `rostopic hz` on the camera topic.)
=============================================================================
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped   # v (forward) + omega (turn)


class LaneFollower:

    WINDOW = "lane follow (debug)"
    SHOW_DEBUG = True          # set False to run headless / no -X needed
    DEBUG_SCALE = 0.6          # debug window size only; no effect on control

    def __init__(self):
        rospy.init_node("lane_follow", anonymous=False)

        # Vehicle name -> used to build the namespaced topic names below.
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # ---- geometry (MUST match panels.py so your tuning carries over) ---
        self.WORK_W, self.WORK_H = 640, 480      # resize every frame to this
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))    # ignore top 50%
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.35))  # row we aim at

        # ---- HSV colour thresholds (copied from your pipeline) -------------
        # If detection fails in your lighting, THESE are what to retune.
        self.yellow_lo = np.array([20, 70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo = np.array([0, 0, 180], dtype=np.uint8)
        self.white_hi = np.array([179, 55, 255], dtype=np.uint8)
        self.min_yellow_area = 30     # ignore yellow blobs smaller than this
        self.min_white_area = 150     # ignore white blobs smaller than this
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ema_alpha = 0.4          # smoothing: higher = snappier, jumpier

        # ---- control parameters (THESE are what you tune for driving) ------
        self.v_nominal = float(rospy.get_param("~v_nominal", 0.22))   # forward speed, m/s
        self.kp = float(rospy.get_param("~kp", 2.5))   # P: how hard it corrects
        self.ki = float(rospy.get_param("~ki", 0.0))   # I: fixes steady drift (leave 0 first)
        self.kd = float(rospy.get_param("~kd", 0.10))  # D: damps wobble/oscillation
        self.omega_max = float(rospy.get_param("~omega_max", 4.0))    # max turn rate, rad/s
        self.deadband_norm = float(rospy.get_param("~deadband_norm", 0.02))  # ignore tiny error
        self.control_rate = float(rospy.get_param("~control_rate", 15.0))    # loop Hz
        self.lost_timeout = float(rospy.get_param("~lost_timeout", 1.0))     # s before STOP
        # Set to -1.0 if the bot steers the WRONG way (turns toward the line
        # it should move away from). Don't touch the math -- just flip this.
        self.steer_sign = float(rospy.get_param("~steer_sign", 1.0))

        # ---- state kept between frames -------------------------------------
        self.yellow_line = None        # last good [vx, vy, x0, y0] fit
        self.white_line = None
        self.smooth_center_x = None    # EMA-smoothed lane centre
        self.err_norm = 0.0            # latest error, normalised to ~[-1, 1]
        self.err_prev = 0.0            # previous error (for the D term)
        self.err_integral = 0.0        # running sum (for the I term)
        self.last_seen_time = rospy.Time.now()   # last time we saw ANY line
        self.latest = None             # newest raw frame from the camera
        self._debug_ok = self.SHOW_DEBUG

        # ---- ROS plumbing --------------------------------------------------
        self.bridge = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)

        # >>> NOT DRIVING #1 <<<  This is the topic the wheel commands go to.
        # If the bot won't move even though this node publishes nonzero v,
        # the problem is almost certainly HERE (wrong topic or switch in
        # manual mode). Verify the real topic with:  rostopic list | grep cmd
        # and override without editing code:  rosrun ... _car_cmd_topic:=/...
        self.cmd_topic = rospy.get_param(
            "~car_cmd_topic", "/{}/car_cmd_switch_node/cmd".format(self.veh))

        # Subscriber: small buff_size so ROS drops stale frames (low latency).
        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.image_callback,
            queue_size=1, buff_size=2 ** 22)
        # Publisher: this is what actually drives the wheels.
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)

        # SAFETY: whenever this node exits, send zero so the bot halts.
        rospy.on_shutdown(self.stop)

        rospy.loginfo("[lane_follow] vehicle = %s", self.veh)
        rospy.loginfo("[lane_follow] camera   = %s", self.cam_topic)
        rospy.loginfo("[lane_follow] commands = %s", self.cmd_topic)
        rospy.loginfo("[lane_follow] v=%.2f kp=%.2f ki=%.2f kd=%.2f",
                      self.v_nominal, self.kp, self.ki, self.kd)

    # ----------------------------------------------------------------------- #
    #  CV helpers (identical to panels.py)
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _find_contours(mask):
        # [-2] makes this work on both OpenCV 3 and 4 (their return tuples differ).
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    @staticmethod
    def _centroid(contour):
        # Centre of mass of a blob; None if the blob has zero area.
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

    @staticmethod
    def _x_at_y(line, y):
        # Given a fitted line and a row y, return the x where the line sits.
        vx, vy, x0, y0 = line
        if abs(vy) < 1e-6:          # near-horizontal guard (avoid divide-by-0)
            return int(x0)
        return int(x0 + (float(y) - float(y0)) * (vx / vy))

    def _clean_mask(self, mask):
        # Open removes specks, close fills small gaps -> cleaner blobs.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    # ----------------------------------------------------------------------- #
    #  Camera callback: keep it CHEAP -- just decode and store newest frame.
    #  (All heavy work happens in the control loop so frames don't queue up.)
    # ----------------------------------------------------------------------- #
    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[lane_follow] decode failed: %s", exc)
            return
        self.latest = cv2.resize(frame, (self.WORK_W, self.WORK_H))

    # ----------------------------------------------------------------------- #
    #  Vision: turn one frame into a single number -- the normalised error.
    #  +error => lane centre is to the RIGHT of the bot (bot drifted left).
    # ----------------------------------------------------------------------- #
    def compute_error(self, frame):
        h, w = frame.shape[:2]

        # Crop to the lower part of the image (the road just ahead).
        roi_y1 = int(h * self.roi_top_ratio)
        roi = frame[roi_y1:h, 0:w]
        rh, rw = roi.shape[:2]
        img_center_x = rw // 2          # where the bot itself is pointing

        # First-frame defaults so the line projections never crash.
        if self.yellow_line is None:
            self.yellow_line = np.array([0.0, 1.0, rw * 0.25, rh * 0.5], dtype=np.float32)
        if self.white_line is None:
            self.white_line = np.array([0.0, 1.0, rw * 0.75, rh * 0.5], dtype=np.float32)
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_center_x)

        # Colour-segment the ROI into yellow and white masks.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask = self._clean_mask(cv2.inRange(hsv, self.white_lo, self.white_hi))

        # LEFT lane = yellow dashes: collect blob centroids and fit a line.
        yellow_centroids = []
        for c in self._find_contours(yellow_mask):
            if cv2.contourArea(c) >= self.min_yellow_area:
                cen = self._centroid(c)
                if cen is not None:
                    yellow_centroids.append(cen)
        yellow_seen = len(yellow_centroids) >= 2     # need >=2 dashes to fit
        if yellow_seen:
            pts = np.array(yellow_centroids, dtype=np.float32)
            self.yellow_line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        # RIGHT lane = solid white: fit a line through the largest white blob.
        white_contours = [c for c in self._find_contours(white_mask)
                          if cv2.contourArea(c) >= self.min_white_area]
        white_seen = len(white_contours) > 0
        if white_seen:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        # Lane centre = midpoint of the two lines at the lookahead row.
        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line, look_y)
        lane_center_x = (xl + xr) / 2.0

        # Smooth it so the steering doesn't jitter frame-to-frame.
        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = float(np.clip(self.smooth_center_x, 0, rw - 1))

        # Normalise: divide pixel offset by half-width -> roughly [-1, 1].
        err_norm = (target_x - img_center_x) / float(img_center_x)

        # >>> NOT DRIVING #2 <<<  We only refresh "last seen" if we actually
        # detected a line this frame. If detection keeps failing, this stops
        # updating, the lost-lane check below fires, and the bot is commanded
        # to STOP. If `rostopic echo` shows v: 0.0 forever, look HERE: your
        # masks aren't finding the lines (retune the HSV thresholds above).
        if yellow_seen or white_seen:
            self.last_seen_time = rospy.Time.now()

        return err_norm, roi, look_y, int(target_x), img_center_x

    # ----------------------------------------------------------------------- #
    #  Optional debug window (lane lines + steering arrow + v/omega readout).
    #  Returns True if the user pressed 'q'. Auto-disables if there's no display.
    # ----------------------------------------------------------------------- #
    def show_debug(self, roi, look_y, target_x, img_center_x, v, omega):
        if not self._debug_ok:
            return False
        rh, rw = roi.shape[:2]
        dbg = roi.copy()
        cv2.line(dbg, (self._x_at_y(self.yellow_line, 0), 0),
                 (self._x_at_y(self.yellow_line, rh - 1), rh - 1), (0, 200, 200), 1)
        cv2.line(dbg, (self._x_at_y(self.white_line, 0), 0),
                 (self._x_at_y(self.white_line, rh - 1), rh - 1), (200, 200, 200), 1)
        cv2.line(dbg, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
        cv2.arrowedLine(dbg, (img_center_x, rh - 1), (target_x, look_y),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.putText(dbg, "v={:.2f} w={:+.2f}".format(v, omega),
                    (4, rh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.DEBUG_SCALE != 1.0:
            dbg = cv2.resize(dbg, None, fx=self.DEBUG_SCALE, fy=self.DEBUG_SCALE,
                             interpolation=cv2.INTER_NEAREST)
        try:
            cv2.imshow(self.WINDOW, dbg)
            return (cv2.waitKey(1) & 0xFF) == ord('q')
        except cv2.error:
            rospy.logwarn("[lane_follow] no display; disabling debug window")
            self._debug_ok = False
        return False

    # ----------------------------------------------------------------------- #
    #  Control loop: runs at control_rate Hz, always on the newest frame.
    # ----------------------------------------------------------------------- #
    def run(self):
        if self._debug_ok:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        rate = rospy.Rate(self.control_rate)
        dt = 1.0 / self.control_rate     # time per loop, for the I and D terms

        while not rospy.is_shutdown():
            frame = self.latest

            # >>> NOT DRIVING #3 <<<  If no camera frame has arrived yet,
            # we loop here forever and NEVER publish. If the bot is silent
            # (rostopic echo shows nothing), the camera topic isn't reaching
            # us -> check VEHICLE_NAME and `rostopic hz` on the camera.
            if frame is None:
                rate.sleep()
                continue

            # Vision -> single error number.
            err_norm, roi, look_y, target_x, img_center_x = self.compute_error(frame)

            # Lost-lane safety: if we haven't seen a line in a while, STOP.
            since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()
            if since_seen > self.lost_timeout:
                self.publish_cmd(0.0, 0.0)      # halt
                self.err_integral = 0.0         # reset I so it doesn't wind up
                if self.show_debug(roi, look_y, target_x, img_center_x, 0.0, 0.0):
                    break
                rate.sleep()
                continue

            # Ignore tiny errors so the bot doesn't twitch when basically centred.
            e = err_norm
            if abs(e) < self.deadband_norm:
                e = 0.0

            # ---- PID controller -------------------------------------------
            self.err_integral = float(np.clip(self.err_integral + e * dt, -1.0, 1.0))
            de = (e - self.err_prev) / dt      # rate of change of the error
            self.err_prev = e
            control = self.kp * e + self.ki * self.err_integral + self.kd * de

            # Sign: +e (lane to the right) -> we must turn RIGHT -> omega < 0.
            # steer_sign flips everything at once if it's backwards.
            omega = float(np.clip(-self.steer_sign * control,
                                  -self.omega_max, self.omega_max))

            # Slow down in sharp turns (big error) for stability.
            v = self.v_nominal * (1.0 - 0.6 * min(abs(e), 1.0))

            # This is the line that actually moves the bot.
            self.publish_cmd(v, omega)

            if self.show_debug(roi, look_y, target_x, img_center_x, v, omega):
                break
            rate.sleep()

    def publish_cmd(self, v, omega):
        # Build and send a Twist2DStamped: v = forward m/s, omega = turn rad/s.
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self.pub_cmd.publish(msg)

    def stop(self):
        # Send several zeros so the halt definitely lands, then clean up.
        for _ in range(10):
            self.publish_cmd(0.0, 0.0)
            rospy.sleep(0.02)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        rospy.loginfo("[lane_follow] stopped.")


def main():
    node = LaneFollower()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.stop()      # belt-and-suspenders: stop on any exit path


if __name__ == "__main__":
    main()



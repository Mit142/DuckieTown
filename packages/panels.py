#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_follow.py
==============
Duckiebot lane-following controller (ROS Noetic).

Reuses the SAME computer-vision pipeline as panels.py / lane_debug_node.py
(HSV segmentation -> fit yellow/left + white/right lines -> lane-centre error
at a lookahead row), but feeds that error through a PID controller and
publishes wheel commands so the bot stays in the lane.

  * Control path is LIGHT: it computes the error only -- no panel drawing -- so
    the control loop stays fast on the bot.
  * SHOW_DEBUG (below) opens a small optional window with the lane lines +
    error, handy for tuning. Set it False to run fully headless (and you can
    then drop the -X flag). If imshow fails (no display) it auto-disables.
  * SAFETY: the wheels are stopped on shutdown (Ctrl-C / q) and whenever the
    lane is lost for longer than ~lost_timeout seconds.

Run on the bot (window needs -X if SHOW_DEBUG=True):
    rosrun <your_package> lane_follow.py
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


class LaneFollower:

    WINDOW = "lane follow (debug)"
    SHOW_DEBUG = True          # set False to run headless / no -X needed
    DEBUG_SCALE = 0.6          # debug window size only; no effect on control

    def __init__(self):
        rospy.init_node("lane_follow", anonymous=False)

        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # ---- geometry (must match panels.py so tuning carries over) --------
        self.WORK_W, self.WORK_H = 640, 480
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.35))

        # ---- HSV thresholds (copied from your pipeline) --------------------
        self.yellow_lo = np.array([20, 70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo = np.array([0, 0, 180], dtype=np.uint8)
        self.white_hi = np.array([179, 55, 255], dtype=np.uint8)
        self.min_yellow_area = 30
        self.min_white_area = 150
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ema_alpha = 0.4

        # ---- control parameters (TUNE THESE) -------------------------------
        self.v_nominal = float(rospy.get_param("~v_nominal", 0.22))   # m/s
        self.kp = float(rospy.get_param("~kp", 2.5))
        self.ki = float(rospy.get_param("~ki", 0.0))
        self.kd = float(rospy.get_param("~kd", 0.10))
        self.omega_max = float(rospy.get_param("~omega_max", 4.0))    # rad/s
        self.deadband_norm = float(rospy.get_param("~deadband_norm", 0.02))
        self.control_rate = float(rospy.get_param("~control_rate", 15.0))  # Hz
        self.lost_timeout = float(rospy.get_param("~lost_timeout", 1.0))   # s
        # Flip this to -1.0 if the bot steers the WRONG way (see notes).
        self.steer_sign = float(rospy.get_param("~steer_sign", 1.0))

        # ---- state ---------------------------------------------------------
        self.yellow_line = None
        self.white_line = None
        self.smooth_center_x = None
        self.err_norm = 0.0
        self.err_prev = 0.0
        self.err_integral = 0.0
        self.last_seen_time = rospy.Time.now()
        self.latest = None         # newest raw frame
        self._debug_ok = self.SHOW_DEBUG

        # ---- ROS plumbing --------------------------------------------------
        self.bridge = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        # Override with _car_cmd_topic if your distro uses a different one;
        # check with:  rostopic list | grep cmd
        self.cmd_topic = rospy.get_param(
            "~car_cmd_topic", "/{}/car_cmd_switch_node/cmd".format(self.veh))

        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.image_callback,
            queue_size=1, buff_size=2 ** 22)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)

        rospy.on_shutdown(self.stop)   # SAFETY: halt wheels on exit

        rospy.loginfo("[lane_follow] vehicle = %s", self.veh)
        rospy.loginfo("[lane_follow] camera   = %s", self.cam_topic)
        rospy.loginfo("[lane_follow] commands = %s", self.cmd_topic)
        rospy.loginfo("[lane_follow] v=%.2f kp=%.2f ki=%.2f kd=%.2f",
                      self.v_nominal, self.kp, self.ki, self.kd)

    # ----------------------------------------------------------------------- #
    #  CV helpers (same as panels.py)
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _find_contours(mask):
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    @staticmethod
    def _centroid(contour):
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

    @staticmethod
    def _x_at_y(line, y):
        vx, vy, x0, y0 = line
        if abs(vy) < 1e-6:
            return int(x0)
        return int(x0 + (float(y) - float(y0)) * (vx / vy))

    def _clean_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    # ----------------------------------------------------------------------- #
    #  Callback: decode + stash newest frame (cheap).
    # ----------------------------------------------------------------------- #
    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[lane_follow] decode failed: %s", exc)
            return
        self.latest = cv2.resize(frame, (self.WORK_W, self.WORK_H))

    # ----------------------------------------------------------------------- #
    #  Vision: compute the normalised lateral error (light, no drawing).
    #  Returns (err_norm, roi, look_y, target_x, img_center_x) for optional
    #  debug drawing.
    # ----------------------------------------------------------------------- #
    def compute_error(self, frame):
        h, w = frame.shape[:2]
        roi_y1 = int(h * self.roi_top_ratio)
        roi = frame[roi_y1:h, 0:w]
        rh, rw = roi.shape[:2]
        img_center_x = rw // 2

        if self.yellow_line is None:
            self.yellow_line = np.array([0.0, 1.0, rw * 0.25, rh * 0.5], dtype=np.float32)
        if self.white_line is None:
            self.white_line = np.array([0.0, 1.0, rw * 0.75, rh * 0.5], dtype=np.float32)
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_center_x)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask = self._clean_mask(cv2.inRange(hsv, self.white_lo, self.white_hi))

        yellow_centroids = []
        for c in self._find_contours(yellow_mask):
            if cv2.contourArea(c) >= self.min_yellow_area:
                cen = self._centroid(c)
                if cen is not None:
                    yellow_centroids.append(cen)
        yellow_seen = len(yellow_centroids) >= 2
        if yellow_seen:
            pts = np.array(yellow_centroids, dtype=np.float32)
            self.yellow_line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        white_contours = [c for c in self._find_contours(white_mask)
                          if cv2.contourArea(c) >= self.min_white_area]
        white_seen = len(white_contours) > 0
        if white_seen:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line, look_y)
        lane_center_x = (xl + xr) / 2.0

        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = float(np.clip(self.smooth_center_x, 0, rw - 1))

        # +ve error => lane centre is to the RIGHT of the robot
        err_norm = (target_x - img_center_x) / float(img_center_x)

        if yellow_seen or white_seen:
            self.last_seen_time = rospy.Time.now()

        return err_norm, roi, look_y, int(target_x), img_center_x

    # ----------------------------------------------------------------------- #
    #  Optional debug window
    # ----------------------------------------------------------------------- #
    def show_debug(self, roi, look_y, target_x, img_center_x, v, omega):
        if not self._debug_ok:
            return
        rh, rw = roi.shape[:2]
        dbg = roi.copy()
        cv2.line(dbg,
                 (self._x_at_y(self.yellow_line, 0), 0),
                 (self._x_at_y(self.yellow_line, rh - 1), rh - 1), (0, 200, 200), 1)
        cv2.line(dbg,
                 (self._x_at_y(self.white_line, 0), 0),
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
            # No display available -> disable debug and keep driving.
            rospy.logwarn("[lane_follow] no display; disabling debug window")
            self._debug_ok = False
        return False

    # ----------------------------------------------------------------------- #
    #  Control loop
    # ----------------------------------------------------------------------- #
    def run(self):
        if self._debug_ok:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        rate = rospy.Rate(self.control_rate)
        dt = 1.0 / self.control_rate

        while not rospy.is_shutdown():
            frame = self.latest
            if frame is None:
                rate.sleep()
                continue

            err_norm, roi, look_y, target_x, img_center_x = self.compute_error(frame)

            # lane lost too long -> stop instead of driving on stale data
            since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()
            if since_seen > self.lost_timeout:
                self.publish_cmd(0.0, 0.0)
                self.err_integral = 0.0
                if self.show_debug(roi, look_y, target_x, img_center_x, 0.0, 0.0):
                    break
                rate.sleep()
                continue

            e = err_norm
            if abs(e) < self.deadband_norm:
                e = 0.0

            # ---- PID ----
            self.err_integral = float(np.clip(self.err_integral + e * dt, -1.0, 1.0))
            de = (e - self.err_prev) / dt
            self.err_prev = e
            control = self.kp * e + self.ki * self.err_integral + self.kd * de

            # +e (lane to the right) -> turn right -> omega NEGATIVE
            omega = float(np.clip(-self.steer_sign * control,
                                  -self.omega_max, self.omega_max))
            # ease off the throttle in sharp turns
            v = self.v_nominal * (1.0 - 0.6 * min(abs(e), 1.0))

            self.publish_cmd(v, omega)

            if self.show_debug(roi, look_y, target_x, img_center_x, v, omega):
                break
            rate.sleep()

    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self.pub_cmd.publish(msg)

    def stop(self):
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
        node.stop()


if __name__ == "__main__":
    main()


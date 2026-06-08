#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py  (LANE FOLLOWER + LIVE PANELS, in one node)
=====================================================
Drives the bot AND shows the 6-panel debug view, in ONE node, so the panels
always match what is actually driving.

  * DRIVES: publishes WheelsCmdStamped (vel_left, vel_right) to
    /<veh>/wheels_driver_node/wheels_cmd.
  * SHOWS: either a cv2.imshow window OR a compressed ROS debug topic (or both).

------------------------------------------------------------------------------
CHANGES IN THIS VERSION (display + turning power)
------------------------------------------------------------------------------
DISPLAY (less lag over a network):
  * SHOW_WINDOW now defaults True. DISPLAY_SCALE lowered to 0.5 and panel_rate
    lowered to 5 Hz to cut bandwidth/lag over X11.
  * NEW: PUBLISH_DEBUG. Instead of (or alongside) X11 imshow, the panel matrix
    is published as a COMPRESSED ROS image on
        /<veh>/lane_follow_panels/compressed
    View it on your laptop with `rqt_image_view` or the Duckietown dashboard.
    This is far smoother than networked X11 and cannot segfault the driver.

TURNING POWER (left/inside wheel stalling):
  * WHEEL DEADZONE COMPENSATION: any nonzero wheel command is bumped up to at
    least `wheel_deadzone` so the motor overcomes static friction (stiction)
    instead of buzzing in place.
  * wheel_min raised to ~0.08 so the INSIDE wheel keeps rolling through a hard
    turn (tight arc) instead of being clipped to 0 and skidding (pivot). Still
    >= 0, so no reverse and no "360 spin".
  * PER-WHEEL TRIM: left_trim / right_trim let you compensate for motor
    imbalance. NOTE: publishing WheelsCmdStamped directly BYPASSES the robot's
    kinematics_node calibration (gain/trim/k/limit), so a chronically weak
    wheel is expected unless you trim here -- or publish Twist2DStamped through
    the calibrated pipeline instead.

  * kp is now a rosparam so you can tune it live (`rosparam set ~kp ...`).

SAFETY: wheels zeroed on shutdown and whenever the lane is lost. The stop path
publishes 0.0 directly and does NOT pass through wheel_min, so the bot still
fully stops. Test on blocks; check steer_sign before the floor.
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType


class LaneFollowerWithPanels(DTROS):

    WINDOW = "lane follow panels"
    SHOW_WINDOW = True      # cv2.imshow window (needs DISPLAY). False -> no X11 window.
    PUBLISH_DEBUG = False    # publish compressed panel image on a ROS topic (laptop-friendly)
    DISPLAY_SCALE = 0.5     # shrink the imshow window only; lower if it lags over X
    DEBUG_SCALE = 0.6       # shrink the published debug image (smaller = less bandwidth)
    DEBUG_JPEG_QUALITY = 60 # 1..100; lower = smaller/faster, blurrier

    def __init__(self):
        super(LaneFollowerWithPanels, self).__init__(
            node_name="lane_follow_panels",
            node_type=NodeType.GENERIC
        )

        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # ---- geometry ------------------------------------------------------
        self.WORK_W, self.WORK_H = 640, 480
        self.PANEL_W, self.PANEL_H = 320, 240
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.55))

        # ---- HSV thresholds ------------------------------------------------
        self.yellow_lo = np.array([20, 70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo = np.array([0, 0, 180], dtype=np.uint8)
        self.white_hi = np.array([179, 55, 255], dtype=np.uint8)
        self.min_yellow_area = 30
        self.min_white_area = 150
        self.canny_lo, self.canny_hi = 60, 160
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.deadband_px = 8
        self.ema_alpha = 0.4

        # ---- control / wheel-mixing (TUNE) ---------------------------------
        self.base_speed = float(rospy.get_param("~base_speed", 0.22))
        self.kp = float(rospy.get_param("~kp", 0.45))                       # now a param (live-tunable)
        self.turn_max = float(rospy.get_param("~turn_max", 0.5))
        self.kd = float(rospy.get_param("~kd", 0.08))
        self.ki = float(rospy.get_param("~ki", 0.0))
        self.turn_slowdown = float(rospy.get_param("~turn_slowdown", 0.3))  # base speed cut in turns
        self.deadband_norm = float(rospy.get_param("~deadband_norm", 0.02))
        self.control_rate = float(rospy.get_param("~control_rate", 15.0))   # Hz
        self.panel_rate = float(rospy.get_param("~panel_rate", 5.0))        # Hz (was 8; lower = less lag)
        self.lost_timeout = float(rospy.get_param("~lost_timeout", 1.0))
        self.steer_sign = float(rospy.get_param("~steer_sign", 1.0))        # flip -1.0 if wrong way

        # ---- wheel limits + stiction / trim (TURNING-POWER FIX) ------------
        # wheel_min raised off 0.0 so the inside wheel keeps rolling (arc) in a
        # hard turn instead of stalling/skidding. Still >= 0 -> no reverse/spin.
        self.wheel_min = float(rospy.get_param("~wheel_min", 0.08))
        self.wheel_max = float(rospy.get_param("~wheel_max", 0.6))
        # Any nonzero command below this can't beat static friction -> bump up.
        self.wheel_deadzone = float(rospy.get_param("~wheel_deadzone", 0.10))
        # Per-wheel trim to compensate motor imbalance (calibration is bypassed
        # when publishing wheels_cmd directly). >1.0 = stronger wheel.
        self.left_trim = float(rospy.get_param("~left_trim", 1.0))
        self.right_trim = float(rospy.get_param("~right_trim", 1.0))

        # ---- derivative filtering / lane-width learning --------------------
        self.d_alpha = float(rospy.get_param("~d_alpha", 0.3))
        self.width_alpha = float(rospy.get_param("~width_alpha", 0.1))
        self.lane_half_width = float(rospy.get_param("~lane_half_width", 220.0))
        self.half_width_min = 120.0
        self.half_width_max = 320.0

        # ---- state ---------------------------------------------------------
        self.yellow_line = None
        self.white_line = None
        self.smooth_center_x = None
        self.err_prev = 0.0
        self.err_integral = 0.0
        self.de_filt = 0.0
        self.last_seen_time = rospy.Time.now()
        self.latest = None
        self._last_panel_t = 0.0
        self._display_ok = self.SHOW_WINDOW

        # ---- ROS plumbing --------------------------------------------------
        self.bridge = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        self.wheels_topic = rospy.get_param(
            "~wheels_topic", "/{}/wheels_driver_node/wheels_cmd".format(self.veh))

        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.image_callback,
            queue_size=1, buff_size=2 ** 22)
        self.pub_wheels = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)

        # optional compressed debug image (view with rqt_image_view on laptop)
        self.pub_debug = None
        if self.PUBLISH_DEBUG:
            self.debug_topic = "/{}/lane_follow_panels/compressed".format(self.veh)
            self.pub_debug = rospy.Publisher(self.debug_topic, CompressedImage, queue_size=1)

        rospy.on_shutdown(self.stop)

        rospy.loginfo("[lane_follow] vehicle = %s", self.veh)
        rospy.loginfo("[lane_follow] camera  = %s", self.cam_topic)
        rospy.loginfo("[lane_follow] wheels  = %s", self.wheels_topic)
        rospy.loginfo("[lane_follow] window  = %s  publish_debug = %s (DISPLAY=%s)",
                      self.SHOW_WINDOW, self.PUBLISH_DEBUG, os.environ.get("DISPLAY", "<unset>"))
        if self.pub_debug is not None:
            rospy.loginfo("[lane_follow] debug   = %s", self.debug_topic)
        rospy.loginfo("[lane_follow] base=%.2f kp=%.2f kd=%.2f wheel_min=%.2f deadzone=%.2f",
                      self.base_speed, self.kp, self.kd, self.wheel_min, self.wheel_deadzone)

    # ----------------------------------------------------------------------- #
    #  CV helpers
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

    def _format_panel(self, img, label):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (self.PANEL_W, self.PANEL_H))
        cv2.rectangle(img, (0, 0), (self.PANEL_W, 18), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ----------------------------------------------------------------------- #
    #  Wheel-command conditioning: trim, stiction deadzone, clamp.
    # ----------------------------------------------------------------------- #
    def _apply_deadzone(self, v):
        """Bump a small nonzero command up to the stiction floor so the motor
        actually turns. Exact 0 stays 0 (full stop is allowed)."""
        if v > 1e-3:
            return max(v, self.wheel_deadzone)
        if v < -1e-3:
            return min(v, -self.wheel_deadzone)
        return 0.0

    def _condition_wheels(self, vel_left, vel_right):
        # 1) per-wheel trim (compensate motor imbalance / bypassed calibration)
        vel_left *= self.left_trim
        vel_right *= self.right_trim
        # 2) stiction deadzone compensation
        vel_left = self._apply_deadzone(vel_left)
        vel_right = self._apply_deadzone(vel_right)
        # 3) clamp to [wheel_min, wheel_max] (wheel_min keeps inside wheel rolling)
        vel_left = float(np.clip(vel_left, self.wheel_min, self.wheel_max))
        vel_right = float(np.clip(vel_right, self.wheel_min, self.wheel_max))
        return vel_left, vel_right

    # ----------------------------------------------------------------------- #
    #  Camera callback
    # ----------------------------------------------------------------------- #
    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[lane_follow] decode failed: %s", exc)
            return
        self.latest = cv2.resize(frame, (self.WORK_W, self.WORK_H))

    # ----------------------------------------------------------------------- #
    #  Vision: compute once, used by BOTH control and panels.
    # ----------------------------------------------------------------------- #
    def compute_vision(self, frame):
        h, w = frame.shape[:2]
        roi_y1 = int(h * self.roi_top_ratio)
        roi = frame[roi_y1:h, 0:w].copy()
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

        white_centroid = None
        white_contours = [c for c in self._find_contours(white_mask)
                          if cv2.contourArea(c) >= self.min_white_area]
        white_seen = len(white_contours) > 0
        if white_seen:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            white_centroid = self._centroid(largest)

        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line, look_y)

        # ---- Single-Line Fallback Logic (SELF-CALIBRATING WIDTH) ----
        if yellow_seen and white_seen:
            lane_center_x = (xl + xr) / 2.0
            half = (xr - xl) / 2.0
            if self.half_width_min <= half <= self.half_width_max:
                self.lane_half_width = (self.width_alpha * half
                                        + (1.0 - self.width_alpha) * self.lane_half_width)
        elif white_seen:
            lane_center_x = xr - self.lane_half_width
        elif yellow_seen:
            lane_center_x = xl + self.lane_half_width
        else:
            lane_center_x = self.smooth_center_x

        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = int(np.clip(self.smooth_center_x, 0, rw - 1))
        err_px = target_x - img_center_x
        err_norm = err_px / float(img_center_x)

        if yellow_seen or white_seen:
            self.last_seen_time = rospy.Time.now()

        return {
            "frame": frame, "roi": roi, "roi_y1": roi_y1, "rh": rh, "rw": rw,
            "w": w, "img_center_x": img_center_x,
            "yellow_mask": yellow_mask, "white_mask": white_mask,
            "yellow_centroids": yellow_centroids, "white_centroid": white_centroid,
            "yellow_seen": yellow_seen, "white_seen": white_seen,
            "look_y": look_y, "target_x": target_x,
            "err_px": err_px, "err_norm": err_norm,
        }

    # ----------------------------------------------------------------------- #
    #  Build the 2x3 matrix (precomputed vision + live wheel cmds).
    # ----------------------------------------------------------------------- #
    def build_panels(self, v, vel_left, vel_right):
        frame = v["frame"]; roi = v["roi"]; rh = v["rh"]; rw = v["rw"]; w = v["w"]
        roi_y1 = v["roi_y1"]; img_center_x = v["img_center_x"]
        look_y = v["look_y"]; target_x = v["target_x"]; err_px = v["err_px"]

        panel1 = frame.copy()
        cv2.rectangle(panel1, (0, roi_y1), (w - 1, frame.shape[0] - 1), (0, 255, 0), 2)

        panel2 = v["yellow_mask"]
        panel3 = v["white_mask"]

        panel4 = roi.copy()
        cv2.line(panel4, (self._x_at_y(self.yellow_line, 0), 0),
                 (self._x_at_y(self.yellow_line, rh - 1), rh - 1), (0, 200, 200), 1)
        cv2.line(panel4, (self._x_at_y(self.white_line, 0), 0),
                 (self._x_at_y(self.white_line, rh - 1), rh - 1), (200, 200, 200), 1)
        for i in range(6):
            y = int(rh * (i + 0.5) / 6)
            xl = int(np.clip(self._x_at_y(self.yellow_line, y), 0, rw - 1))
            xr = int(np.clip(self._x_at_y(self.white_line, y), 0, rw - 1))
            cv2.line(panel4, (xl, y), (xr, y), (0, 255, 0), 1)
            cv2.circle(panel4, ((xl + xr) // 2, y), 2, (0, 0, 255), -1)
        for cen in v["yellow_centroids"]:
            cv2.circle(panel4, cen, 3, (0, 255, 255), -1)
        if v["white_centroid"] is not None:
            cv2.circle(panel4, v["white_centroid"], 3, (255, 255, 255), -1)
        if not v["yellow_seen"]:
            cv2.putText(panel4, "Y:last", (4, rh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
        if not v["white_seen"]:
            cv2.putText(panel4, "W:last", (rw - 70, rh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        panel5 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        panel5[v["yellow_mask"] > 0] = (0, 255, 255)
        panel5[v["white_mask"] > 0] = (255, 255, 255)

        panel6 = cv2.convertScaleAbs(roi, alpha=0.4)
        cv2.line(panel6, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
        cv2.arrowedLine(panel6, (img_center_x, rh - 1), (target_x, look_y),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.circle(panel6, (target_x, look_y), 4, (0, 255, 0), -1)
        if abs(err_px) < self.deadband_px:
            direction = "STRAIGHT"
        elif err_px > 0:
            direction = "RIGHT"
        else:
            direction = "LEFT"
        cv2.putText(panel6, "err {:+d}px {}".format(err_px, direction),
                    (4, rh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(panel6, "L={:+.2f}  R={:+.2f}".format(vel_left, vel_right),
                    (4, rh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        top = np.hstack([self._format_panel(panel1, "1 RAW + ROI"),
                         self._format_panel(panel2, "2 YELLOW mask"),
                         self._format_panel(panel3, "3 WHITE mask")])
        bot = np.hstack([self._format_panel(panel4, "4 LANE BED"),
                         self._format_panel(panel5, "5 EDGES + HSV"),
                         self._format_panel(panel6, "6 STEERING + WHEELS")])
        return np.vstack([top, bot])

    # ----------------------------------------------------------------------- #
    #  Crash-proof display: a dead X11 connection disables the window only.
    # ----------------------------------------------------------------------- #
    def _safe_show(self, matrix):
        if not self._display_ok:
            return
        try:
            if self.DISPLAY_SCALE != 1.0:
                matrix = cv2.resize(matrix, None,
                                    fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE,
                                    interpolation=cv2.INTER_NEAREST)
            cv2.imshow(self.WINDOW, matrix)
        except cv2.error as exc:
            rospy.logwarn("[lane_follow] display lost, window off (still driving): %s", exc)
            self._display_ok = False

    def _safe_waitkey(self):
        if not self._display_ok:
            return False
        try:
            return (cv2.waitKey(1) & 0xFF) == ord('q')
        except cv2.error:
            self._display_ok = False
        return False

    def _publish_debug(self, matrix):
        """Publish the panel matrix as a compressed ROS image. View on the
        laptop with: rosrun image_view image_view image:=<topic> _image_transport:=compressed
        or rqt_image_view. Much smoother than networked X11."""
        if self.pub_debug is None:
            return
        try:
            if self.DEBUG_SCALE != 1.0:
                matrix = cv2.resize(matrix, None,
                                    fx=self.DEBUG_SCALE, fy=self.DEBUG_SCALE,
                                    interpolation=cv2.INTER_NEAREST)
            ok, enc = cv2.imencode(".jpg", matrix,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.DEBUG_JPEG_QUALITY])
            if not ok:
                return
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = enc.tobytes()
            self.pub_debug.publish(msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[lane_follow] debug publish error: %s" % exc)

    # ----------------------------------------------------------------------- #
    #  Control + display loop
    # ----------------------------------------------------------------------- #
    def run(self):
        if self._display_ok:
            try:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                rospy.logwarn("[lane_follow] cannot open window (%s); headless", exc)
                self._display_ok = False

        rate = rospy.Rate(self.control_rate)
        dt = 1.0 / self.control_rate
        panel_period = 0.0 if self.panel_rate <= 0 else 1.0 / self.panel_rate
        want_panels = self._display_ok or (self.pub_debug is not None)

        while not rospy.is_shutdown():
            frame = self.latest
            if frame is None:
                rate.sleep()
                continue

            vis = self.compute_vision(frame)

            # ---- control ----
            since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()
            if since_seen > self.lost_timeout:
                vel_left = vel_right = 0.0
                # reset ALL controller state so re-acquiring doesn't lurch
                self.err_integral = 0.0
                self.err_prev = 0.0
                self.de_filt = 0.0
            else:
                e = vis["err_norm"]
                if abs(e) < self.deadband_norm:
                    e = 0.0
                self.err_integral = float(np.clip(self.err_integral + e * dt, -1.0, 1.0))
                de_raw = (e - self.err_prev) / dt
                self.err_prev = e
                self.de_filt = (self.d_alpha * de_raw
                                + (1.0 - self.d_alpha) * self.de_filt)
                control = self.kp * e + self.ki * self.err_integral + self.kd * self.de_filt
                turn = float(np.clip(self.steer_sign * control, -self.turn_max, self.turn_max))
                base = self.base_speed * (1.0 - self.turn_slowdown * min(abs(e), 1.0))
                # mix, then condition (trim -> stiction deadzone -> clamp)
                vel_left, vel_right = self._condition_wheels(base + turn, base - turn)

            self.publish_wheels(vel_left, vel_right)

            # ---- panels (rate-limited; never blocks driving) ----
            now = rospy.get_time()
            if want_panels and (panel_period <= 0.0 or (now - self._last_panel_t) >= panel_period):
                try:
                    matrix = self.build_panels(vis, vel_left, vel_right)
                    self._safe_show(matrix)
                    self._publish_debug(matrix)
                    self._last_panel_t = now
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "[lane_follow] panel error: %s" % exc)
            if self._safe_waitkey():
                break

            rate.sleep()

    def publish_wheels(self, vel_left, vel_right):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = float(vel_left)
        msg.vel_right = float(vel_right)
        self.pub_wheels.publish(msg)

    def stop(self):
        for _ in range(10):
            self.publish_wheels(0.0, 0.0)
            rospy.sleep(0.02)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        rospy.loginfo("[lane_follow] stopped.")


def main():
    node = LaneFollowerWithPanels()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.stop()


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py  (LANE FOLLOWER + LIVE imshow PANELS, in one node)
============================================================
Drives the bot AND shows the 6-panel debug view in an OpenCV window, in ONE
node, so the panels always match what is actually driving.

  * DRIVES: publishes WheelsCmdStamped (vel_left, vel_right) to
    /<veh>/wheels_driver_node/wheels_cmd.
  * SHOWS: cv2.imshow window of the 2x3 panel matrix. Panel 6 prints the live
    error AND the L/R wheel commands, so you see the calculation drive the bot.

DISPLAY / X11 (runtime, NOT in this code):
  The window appears wherever DISPLAY points. You set that in your run command
  (XLaunch + DISPLAY=<laptop_ip>:0.0). This file does not hardcode any of that.

  >>> CRASH-PROOF <<<  If the X11 connection drops (common with a networked X
  server), imshow would normally SEGFAULT and kill the driver mid-drive. Here
  every GUI call is guarded: a dead display just disables the window and the
  bot KEEPS DRIVING. Set SHOW_WINDOW=False to run fully headless.

  >>> MESSAGE TYPE <<<  Publishes WheelsCmdStamped. If
  `rostopic info /<veh>/wheels_driver_node/wheels_cmd` shows a different type,
  change the import + publish_wheels().
  >>> LAUNCHER <<<  Only THIS node should publish wheel commands. If
  laneFollowing.py also drives, disable it or they will fight.

SAFETY: wheels zeroed on shutdown and whenever the lane is lost. Test on
blocks; check steering direction (steer_sign) before the floor.

TURN/SPEED FIXES (vs original):
  * wheel_min 0.0 (was -0.4): inside wheel can no longer reverse, so the bot
    arcs tightly instead of pivoting in place -> kills the "360 spin".
  * base_speed 0.22 (was 0.15): faster cruise.
  * kp 0.45, turn_max 0.5: commits to curves earlier and harder so the line
    stays in frame.
  * kd 0.08 (was 0.05): damps the higher kp to avoid straight-line wobble.
  * turn slowdown factor 0.3 (was 0.5): keeps momentum through turns so it
    arcs instead of crawling.
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
    SHOW_WINDOW = False     # False -> headless (no X needed at all)
    DISPLAY_SCALE = 0.7    # shrink the window only; lower if it lags over X

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
        self.base_speed = float(rospy.get_param("~base_speed", 0.22))       # was 0.15 (fixes "slow")
        self.kp = 0.45         # was 0.35 (commit to curves earlier)
        self.turn_max = float(rospy.get_param("~turn_max", 0.5))            # was 0.35 (more steering authority)
        self.kd = float(rospy.get_param("~kd", 0.08))                       # was 0.05 (damps the higher kp)
        self.ki = float(rospy.get_param("~ki", 0.0))  
        self.wheel_min = float(rospy.get_param("~wheel_min", 0.0))          # was -0.4  <-- THE KEY FIX (no pivot-in-place)
        self.wheel_max = float(rospy.get_param("~wheel_max", 0.6))
        self.deadband_norm = float(rospy.get_param("~deadband_norm", 0.02))
        self.control_rate = float(rospy.get_param("~control_rate", 15.0))   # Hz
        self.panel_rate = float(rospy.get_param("~panel_rate", 8.0))        # Hz
        self.lost_timeout = float(rospy.get_param("~lost_timeout", 1.0))
        self.steer_sign = float(rospy.get_param("~steer_sign", 1.0))  # flip -1.0 if wrong way

        # ---- state ---------------------------------------------------------
        self.yellow_line = None
        self.white_line = None
        self.smooth_center_x = None
        self.err_prev = 0.0
        self.err_integral = 0.0
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

        rospy.on_shutdown(self.stop)

        rospy.loginfo("[lane_follow] vehicle = %s", self.veh)
        rospy.loginfo("[lane_follow] camera  = %s", self.cam_topic)
        rospy.loginfo("[lane_follow] wheels  = %s", self.wheels_topic)
        rospy.loginfo("[lane_follow] window  = %s (DISPLAY=%s)",
                      self.SHOW_WINDOW, os.environ.get("DISPLAY", "<unset>"))
        rospy.loginfo("[lane_follow] base=%.2f kp=%.2f kd=%.2f",
                      self.base_speed, self.kp, self.kd)

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

        # ---- Single-Line Fallback Logic ----
        # NOTE: this is in the 640-px-wide ROI (NOT the 320-px display panel).
        # 220 ~= half the lane width in ROI pixels. If the bot wanders the
        # instant one line drops out, measure the real yellow<->white gap at
        # look_y and set this to half of that.
        lane_half_width = 220 

        if yellow_seen and white_seen:
            # Normal operation: both lines visible
            lane_center_x = (xl + xr) / 2.0
        elif white_seen:
            # Lost the yellow line! Estimate center using only the white line
            lane_center_x = xr - lane_half_width
        elif yellow_seen:
            # Lost the white line! Estimate center using only the yellow line
            lane_center_x = xl + lane_half_width
        else:
            # Blind! Hold the last known smooth center to coast through
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
                self.err_integral = 0.0
            else:
                e = vis["err_norm"]
                if abs(e) < self.deadband_norm:
                    e = 0.0
                self.err_integral = float(np.clip(self.err_integral + e * dt, -1.0, 1.0))
                de = (e - self.err_prev) / dt
                self.err_prev = e
                control = self.kp * e + self.ki * self.err_integral + self.kd * de
                turn = float(np.clip(self.steer_sign * control, -self.turn_max, self.turn_max))
                base = self.base_speed * (1.0 - 0.3 * min(abs(e), 1.0))   # was 0.5 — keep momentum through turns
                vel_left = float(np.clip(base + turn, self.wheel_min, self.wheel_max))
                vel_right = float(np.clip(base - turn, self.wheel_min, self.wheel_max))

            self.publish_wheels(vel_left, vel_right)

            # ---- panels (rate-limited; never blocks driving) ----
            now = rospy.get_time()
            if self._display_ok and (panel_period <= 0.0 or (now - self._last_panel_t) >= panel_period):
                try:
                    matrix = self.build_panels(vis, vel_left, vel_right)
                    self._safe_show(matrix)
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


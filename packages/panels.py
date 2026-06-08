#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py
=========
DISPLAY + DRIVE variant of the 6-panel lane-debug node (ROS Noetic).

Same computer-vision pipeline + display as before, with the steering reworked
so it stops OVER-ESTIMATING turns and running out of the lane on corners.

    +---------------------+---------------------+---------------------+
    | 1  RAW + ROI box    | 2  YELLOW mask      | 3  WHITE mask       |
    +---------------------+---------------------+---------------------+
    | 4  LANE BED overlay | 5  EDGES / combined | 6  STEERING + v/w   |
    +---------------------+---------------------+---------------------+

TURN FIX (what changed in this version)
---------------------------------------
1. NEARER LOOKAHEAD. lookahead_ratio moved 0.35 -> 0.55, so the lane center is
   measured closer to the bot. We fit a STRAIGHT line; extrapolating it far
   down a CURVED road throws the target way out to the side, which made the
   controller turn early and cut the corner. Measuring nearer kills most of
   that extrapolation error. (Tune 0.45 near .. 0.70 far; bigger = nearer.)
2. NO STALE-LINE AVERAGING. On a curve the dashed yellow drops out (or white
   leaves the ROI). The old code averaged the fresh line with the *last good*
   stale one, yanking the center sideways. Now, if only one line is fresh, the
   center is placed a remembered half-lane-width from THAT line.
3. OMEGA SLEW LIMIT. Steering can't jump more than omega_slew rad/s per second,
   so one noisy frame can't snap into a hard turn.
4. Gentler defaults: kp 3.0 -> 2.2, omega_max 5.0 -> 4.0.

COMMAND TOPIC
-------------
Twist2DStamped to /<veh>/car_cmd_switch_node/cmd (v m/s, omega rad/s).

SAFETY
------
Zero command on quit / Ctrl-C; stops if the lane is lost for lost_timeout s.
Press 'q'.

Run on the bot (with -X so the window forwards to your laptop):
    DOCKER_API_VERSION=1.41 dts devel build -f -H 192.168.3.8
    DOCKER_API_VERSION=1.41 docker -H 192.168.3.8 rm -f dts-run-duckietown 2>/dev/null
    DOCKER_API_VERSION=1.41 dts devel run -H 192.168.3.8 -X --rm
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


class LanePanelsDrive:
    """Builds + shows the 6-panel debug matrix AND drives the lane."""

    WINDOW = "lane panels"
    DISPLAY_SCALE = 1.0   # 1.0 = full 960x480 matrix; lower if the window lags

    def __init__(self):
        rospy.init_node("panels", anonymous=False)

        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # Geometry. WORK_* = working resolution; PANEL_* = each panel.
        self.WORK_W, self.WORK_H = 320, 240  # <--- Lowered from 640, 480
        self.PANEL_W, self.PANEL_H = 160, 120

        # ---- Vision tunables (overridable via the ROS param server) -------
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))
        # NEARER lookahead than before (0.35 -> 0.55): less corner-cutting.
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.55))
        self.display_rate = float(rospy.get_param("~display_rate", 15.0))  # Hz

        # HSV thresholds (OpenCV: H 0-179, S 0-255, V 0-255). Retune for light.
        self.yellow_lo = np.array([20, 70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo = np.array([0, 0, 180], dtype=np.uint8)
        self.white_hi = np.array([179, 55, 255], dtype=np.uint8)

        self.min_yellow_area = 30
        self.min_white_area = 150
        self.canny_lo, self.canny_hi = 60, 160
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.deadband = 8          # px error treated as "straight"
        self.ema_alpha = 0.6       # center smoothing (higher = less lag)

        # ---- Control tunables (THE knobs for your two symptoms) -----------
        self.enable_drive = bool(rospy.get_param("~enable_drive", True))
        self.v_nominal = float(rospy.get_param("~v_nominal", 0.23))   # m/s straight
        self.v_min = float(rospy.get_param("~v_min", 0.08))           # m/s floor
        self.Kp = float(rospy.get_param("~kp", 2.2))    # rad/s per unit error
        self.Kd = float(rospy.get_param("~kd", 0.4))    # damping (kills wobble)
        self.omega_max = float(rospy.get_param("~omega_max", 4.0))    # rad/s clamp
        # max steering change per second (slew). lower = smoother, less snappy.
        self.omega_slew = float(rospy.get_param("~omega_slew", 10.0))  # rad/s^2
        # slow down on sharp turns: at full error, v = v_nominal*(1-turn_slowdown)
        self.turn_slowdown = float(rospy.get_param("~turn_slowdown", 0.6))
        self.d_alpha = float(rospy.get_param("~d_alpha", 0.5))  # derivative LPF
        # +1 or -1 depending on which way your bot steers; flip if it veers
        # AWAY from the lane center instead of toward it.
        self.omega_sign = float(rospy.get_param("~omega_sign", -1.0))
        self.lost_timeout = float(rospy.get_param("~lost_timeout", 0.6))  # s

        # ---- Persistent state ---------------------------------------------
        # cv2.fitLine returns [vx, vy, x0, y0]: unit dir (vx,vy) + point (x0,y0).
        self.yellow_line = None
        self.white_line = None
        self.smooth_center_x = None
        self.lane_half_width = None   # remembered half lane width (px, at look_y)
        self.yellow_seen = False
        self.white_seen = False

        self._prev_e = 0.0
        self._prev_t = None
        self._d_filt = 0.0
        self._prev_omega = 0.0
        self._last_seen_t = None
        self._cmd_v = 0.0
        self._cmd_omega = 0.0

        self.latest = None        # newest raw frame (callback -> main loop)
        self._frames = 0

        # ---- ROS plumbing -------------------------------------------------
        self.bridge = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.image_callback,
            queue_size=1, buff_size=2 ** 22,
        )

        # Standard lane-following command. v=forward(m/s), omega=turn(rad/s).
        # >>> WHEELS_CMD alternative: if your working driver used raw wheel
        #     velocities instead, import WheelsCmdStamped from duckietown_msgs,
        #     publish to /<veh>/wheels_driver_node/wheels_cmd, and convert:
        #         vel_left  = v - 0.5 * BASELINE * omega
        #         vel_right = v + 0.5 * BASELINE * omega
        #     (Twist2DStamped is preferred: calibration is applied for you.)
        self.cmd_topic = "/{}/car_cmd_switch_node/cmd".format(self.veh)
        self.pub = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)

        rospy.on_shutdown(self._publish_stop)

        rospy.loginfo("[panels] vehicle      = %s", self.veh)
        rospy.loginfo("[panels] camera topic = %s", self.cam_topic)
        rospy.loginfo("[panels] cmd topic    = %s", self.cmd_topic)
        rospy.loginfo("[panels] DISPLAY env  = %s", os.environ.get("DISPLAY", "(unset!)"))
        rospy.loginfo("[panels] drive enabled= %s", self.enable_drive)
        rospy.loginfo("[panels] press q in the window to quit")

    # ======================================================================
    #  Helpers
    # ======================================================================
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

    def _placeholder(self, text):
        """A gray 960x480 matrix so the window appears before frames arrive."""
        ph = np.full((2 * self.PANEL_H, 3 * self.PANEL_W, 3), 40, dtype=np.uint8)
        cv2.putText(ph, text, (40, self.PANEL_H), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (200, 200, 200), 2, cv2.LINE_AA)
        return ph

    # ======================================================================
    #  Control: lane-center pixel error -> (v, omega)
    # ======================================================================
    def _compute_control(self, error_px, half_width_px, lane_seen):
        now = rospy.get_time()

        # If we have not seen a line recently, stop instead of driving blind.
        if lane_seen:
            self._last_seen_t = now
        if self._last_seen_t is None or (now - self._last_seen_t) > self.lost_timeout:
            self._prev_e = 0.0
            self._d_filt = 0.0
            self._prev_omega = 0.0
            return 0.0, 0.0

        # Normalize error to roughly [-1, 1] so gains are resolution-agnostic.
        e = float(error_px) / float(max(1, half_width_px))
        e = float(np.clip(e, -1.0, 1.0))

        # Filtered derivative (damps oscillation without amplifying pixel noise).
        dt = (now - self._prev_t) if self._prev_t is not None else 0.0
        self._prev_t = now
        de = (e - self._prev_e) / dt if dt > 1e-3 else 0.0
        self._prev_e = e
        self._d_filt = self.d_alpha * de + (1.0 - self.d_alpha) * self._d_filt

        # PD -> omega. omega_sign flips if the bot steers the wrong way.
        omega = self.omega_sign * (self.Kp * e + self.Kd * self._d_filt)
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))

        # Slew-limit: cap how fast omega can change so a single bad frame
        # cannot snap the steering into a hard turn.
        if dt > 1e-3:
            max_step = self.omega_slew * dt
            omega = float(np.clip(omega,
                                  self._prev_omega - max_step,
                                  self._prev_omega + max_step))
        self._prev_omega = omega

        # Slow down as the turn sharpens so we don't run wide on corners.
        v = self.v_nominal * (1.0 - self.turn_slowdown * min(1.0, abs(e)))
        v = max(self.v_min, v)
        return v, omega

    def _publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self.pub.publish(msg)

    def _publish_stop(self):
        # Called on shutdown and on quit. Send a few zeros to be sure.
        try:
            for _ in range(5):
                self._publish_cmd(0.0, 0.0)
                rospy.sleep(0.02)
        except Exception:
            pass

    # ======================================================================
    #  Callback: just decode + stash the newest frame (cheap).
    # ======================================================================
    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[panels] decode failed: %s", exc)
            return
        self.latest = cv2.resize(frame, (self.WORK_W, self.WORK_H))
        self._frames += 1
        if self._frames == 1:
            rospy.loginfo("[panels] first camera frame received -- pipeline live")

    # ======================================================================
    #  Pipeline -> assembled 2x3 BGR matrix (also sets self._cmd_v/_omega)
    # ======================================================================
    def process_frame(self, frame):
        h, w = frame.shape[:2]

        # ---------- ROI (lower road portion) -------------------------------
        roi_y1 = int(h * self.roi_top_ratio)
        roi_y2 = h
        roi = frame[roi_y1:roi_y2, 0:w].copy()
        rh, rw = roi.shape[:2]
        img_center_x = rw // 2

        if self.yellow_line is None:
            self.yellow_line = np.array([0.0, 1.0, rw * 0.25, rh * 0.5], dtype=np.float32)
        if self.white_line is None:
            self.white_line = np.array([0.0, 1.0, rw * 0.75, rh * 0.5], dtype=np.float32)
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_center_x)
        if self.lane_half_width is None:
            self.lane_half_width = rw * 0.25   # initial guess: quarter-width

        # ---------- HSV segmentation ---------------------------------------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask = self._clean_mask(cv2.inRange(hsv, self.white_lo, self.white_hi))

        # ---------- Left (yellow, dashed): fit line through dash centroids --
        yellow_centroids = []
        for c in self._find_contours(yellow_mask):
            if cv2.contourArea(c) >= self.min_yellow_area:
                cen = self._centroid(c)
                if cen is not None:
                    yellow_centroids.append(cen)

        if len(yellow_centroids) >= 2:
            pts = np.array(yellow_centroids, dtype=np.float32)
            self.yellow_line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            self.yellow_seen = True
        else:
            self.yellow_seen = False  # FALLBACK: reuse last good fit

        # ---------- Right (white, solid): fit line through largest contour --
        white_centroid = None
        white_contours = [c for c in self._find_contours(white_mask)
                          if cv2.contourArea(c) >= self.min_white_area]
        if white_contours:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            white_centroid = self._centroid(largest)
            self.white_seen = True
        else:
            self.white_seen = False  # FALLBACK: reuse last good fit

        # ---------- PANEL 1: raw + ROI box ---------------------------------
        panel1 = frame.copy()
        cv2.rectangle(panel1, (0, roi_y1), (w - 1, roi_y2 - 1), (0, 255, 0), 2)
        cv2.putText(panel1, "ROI", (6, roi_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # ---------- PANELS 2 & 3: colour masks -----------------------------
        panel2 = yellow_mask
        panel3 = white_mask

        # ---------- PANEL 4: lane-bed overlay ------------------------------
        panel4 = roi.copy()
        y_top, y_bot = 0, rh - 1
        cv2.line(panel4,
                 (self._x_at_y(self.yellow_line, y_top), y_top),
                 (self._x_at_y(self.yellow_line, y_bot), y_bot),
                 (0, 200, 200), 1)
        cv2.line(panel4,
                 (self._x_at_y(self.white_line, y_top), y_top),
                 (self._x_at_y(self.white_line, y_bot), y_bot),
                 (200, 200, 200), 1)

        n_bands = 6
        for i in range(n_bands):
            y = int(rh * (i + 0.5) / n_bands)
            xl = int(np.clip(self._x_at_y(self.yellow_line, y), 0, rw - 1))
            xr = int(np.clip(self._x_at_y(self.white_line, y), 0, rw - 1))
            cv2.line(panel4, (xl, y), (xr, y), (0, 255, 0), 1)
            xc = (xl + xr) // 2
            cv2.circle(panel4, (xc, y), 2, (0, 0, 255), -1)

        for cen in yellow_centroids:
            cv2.circle(panel4, cen, 3, (0, 255, 255), -1)
        if white_centroid is not None:
            cv2.circle(panel4, white_centroid, 3, (255, 255, 255), -1)

        if not self.yellow_seen:
            cv2.putText(panel4, "Y:last", (4, rh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
        if not self.white_seen:
            cv2.putText(panel4, "W:last", (rw - 70, rh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ---------- PANEL 5: Canny + colour tint ---------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        panel5 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        panel5[yellow_mask > 0] = (0, 255, 255)
        panel5[white_mask > 0] = (255, 255, 255)

        # ---------- PANEL 6: steering indicator + control ------------------
        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line, look_y)

        # --- Robust lane-center: DON'T average a stale line on a curve. -----
        # Both fresh  -> midpoint, and refresh the remembered half-width.
        # One fresh    -> offset the remembered half-width from THAT line.
        # None fresh   -> hold the last smoothed center (lost-timeout handles
        #                 a sustained loss by stopping the bot).
        if self.yellow_seen and self.white_seen:
            lane_center_x = (xl + xr) / 2.0
            hw = (xr - xl) / 2.0
            if hw > rw * 0.05:   # guard against crossed / degenerate fits
                self.lane_half_width = 0.7 * self.lane_half_width + 0.3 * hw
        elif self.yellow_seen:
            lane_center_x = xl + self.lane_half_width
        elif self.white_seen:
            lane_center_x = xr - self.lane_half_width
        else:
            lane_center_x = self.smooth_center_x

        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = int(np.clip(self.smooth_center_x, 0, rw - 1))
        error = target_x - img_center_x

        # ---- compute the drive command from the same error ----
        lane_seen = self.yellow_seen or self.white_seen
        v, omega = self._compute_control(error, img_center_x, lane_seen)
        self._cmd_v, self._cmd_omega = v, omega

        panel6 = cv2.convertScaleAbs(roi, alpha=0.4)
        bot_origin = (img_center_x, rh - 1)
        cv2.line(panel6, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
        cv2.line(panel6, (0, look_y), (rw - 1, look_y), (120, 120, 120), 1)
        cv2.arrowedLine(panel6, bot_origin, (target_x, look_y),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.circle(panel6, (target_x, look_y), 4, (0, 255, 0), -1)

        if abs(error) < self.deadband:
            direction = "STRAIGHT"
        elif error > 0:
            direction = "RIGHT"
        else:
            direction = "LEFT"
        cv2.putText(panel6, "err {:+d}px {}".format(error, direction),
                    (4, rh - 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(panel6, "v {:.2f}  w {:+.2f}".format(v, omega),
                    (4, rh - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ---------- Assemble 2x3 matrix ------------------------------------
        top_row = np.hstack([
            self._format_panel(panel1, "1 RAW + ROI"),
            self._format_panel(panel2, "2 YELLOW mask"),
            self._format_panel(panel3, "3 WHITE mask"),
        ])
        bottom_row = np.hstack([
            self._format_panel(panel4, "4 LANE BED"),
            self._format_panel(panel5, "5 EDGES + HSV"),
            self._format_panel(panel6, "6 STEERING"),
        ])
        return np.vstack([top_row, bottom_row])

    # ======================================================================
    #  Display helper (so a headless build fails loudly, not silently).
    # ======================================================================
    def _show(self, matrix):
        if self.DISPLAY_SCALE != 1.0:
            matrix = cv2.resize(matrix, None, fx=self.DISPLAY_SCALE,
                                fy=self.DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        try:
            cv2.imshow(self.WINDOW, matrix)
        except cv2.error as exc:
            rospy.logerr_throttle(
                5.0,
                "[panels] imshow failed (%s). This usually means the bot's "
                "OpenCV is the HEADLESS build. Add 'opencv-python' (not "
                "-headless) to dependencies-py3.txt and rebuild." % exc)

    # ======================================================================
    #  Main loop: process newest frame -> show + drive (GUI on main thread).
    # ======================================================================
    def spin(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        # Paint a placeholder immediately so we KNOW the window opened.
        self._show(self._placeholder("waiting for camera..."))
        cv2.waitKey(1)

        rate = rospy.Rate(self.display_rate)
        while not rospy.is_shutdown():
            frame = self.latest
            matrix = None
            if frame is not None:
                try:
                    matrix = self.process_frame(frame)
                except Exception as exc:  # stay alive on a bad frame
                    rospy.logwarn_throttle(2.0, "[panels] pipeline error: %s" % exc)
                    # don't keep last command on a broken frame -> coast to stop
                    self._cmd_v, self._cmd_omega = 0.0, 0.0

            # Drive (publish every loop so the switch keeps getting commands).
            if self.enable_drive:
                self._publish_cmd(self._cmd_v, self._cmd_omega)

            if matrix is not None:
                self._show(matrix)
            else:
                # still pump the GUI so the placeholder window stays responsive
                self._show(self._placeholder("waiting for camera..."))

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

        self._publish_stop()
        cv2.destroyAllWindows()


def main():
    node = LanePanelsDrive()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            node._publish_stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


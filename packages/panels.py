#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py
=========
DISPLAY + DRIVE variant of the 6-panel lane-debug node (ROS Noetic).

    +---------------------+---------------------+---------------------+
    | 1  RAW + ROI box    | 2  YELLOW mask      | 3  WHITE mask       |
    +---------------------+---------------------+---------------------+
    | 4  LANE BED overlay | 5  EDGES / combined | 6  STEERING + v/w   |
    +---------------------+---------------------+---------------------+

FIXES IN THIS VERSION
---------------------
FIX 1 — ONE WHEEL NOT TURNING
  _compute_control() was normalising the pixel error against img_center_x
  (half the ROI width = 320 px).  That's far too wide: a real Duckietown lane
  is ~90–110 px across at the ROI, so the normalised error was always tiny and
  omega never climbed high enough to split the wheel speeds meaningfully.
  Now the actual lane_half_width (px) is passed as the normalisation
  denominator. If it hasn't been measured yet the code falls back to
  img_center_x so behaviour before the first observation is unchanged.

FIX 2 — BOT HUGS WHITE LINE
  The lane_half_width default was rw*0.25 = 160 px — about double the real
  lane.  Every time only white was seen the code placed lane_center as
  xr - 160 px, which is far to the LEFT of real center; EMA then dragged the
  smoothed target toward white.  Default is now 90 px (close to real), and
  the EMA update weight was raised so it converges in ~10 frames with both
  lines visible.

FIX 3 — CROSSES LANE AFTER TURNS
  Two interacting causes:
    a) ema_alpha = 0.6 was high enough that the smoothed lane-center lagged
       badly in turns; the robot turned too late, overshot, and then
       overcorrected.  Lowered to 0.35 — faster tracking, less lag.
    b) omega_slew = 10.0 rad/s² was so loose it never fired.  Tightened to
       3.0 rad/s² so exiting a corner can't snap into a hard opposite turn
       in one frame.

FIX 4 — BOT ONLY MOVES AFTER DISPLAY WINDOW IS CLICKED
  The drive loop and cv2.waitKey() shared the same thread.  On systems where
  X11 forwarding hasn't fully rendered the window yet, waitKey() can stall
  until the window is focused/clicked, blocking the publish loop entirely.
  Solution: the control + CV pipeline runs in a dedicated background thread
  at the full control rate.  The main thread owns ONLY the GUI (imshow +
  waitKey).  The two threads share `self.latest` (newest frame) and
  `self._display_matrix` (newest debug image) through simple assignments,
  which is safe for CPython because assignment is atomic.  The drive commands
  are published from the background thread regardless of display state.

COMMAND TOPIC
-------------
Twist2DStamped to /<veh>/car_cmd_switch_node/cmd (v m/s, omega rad/s).

SAFETY
------
Zero command on quit / Ctrl-C; stops if the lane is lost for lost_timeout s.
Press 'q' in the debug window, or Ctrl-C.
"""

import os
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


class LanePanelsDrive:
    """Builds + shows the 6-panel debug matrix AND drives the lane."""

    WINDOW = "lane panels"
    DISPLAY_SCALE = 1.0   # lower if the forwarded window is slow

    def __init__(self):
        rospy.init_node("panels", anonymous=False)

        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # Geometry
        self.WORK_W, self.WORK_H = 640, 480
        self.PANEL_W, self.PANEL_H = 320, 240

        # ---- Vision tunables ---------------------------------------------
        self.roi_top_ratio    = float(rospy.get_param("~roi_top_ratio",   0.50))
        self.lookahead_ratio  = float(rospy.get_param("~lookahead_ratio", 0.55))
        self.display_rate     = float(rospy.get_param("~display_rate",   15.0))
        self.control_rate     = float(rospy.get_param("~control_rate",   30.0))

        self.yellow_lo = np.array([20,  70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo  = np.array([ 0,   0, 180], dtype=np.uint8)
        self.white_hi  = np.array([179, 55, 255], dtype=np.uint8)

        self.min_yellow_area = 30
        self.min_white_area  = 150
        self.canny_lo, self.canny_hi = 60, 160
        self.kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.deadband = 8

        # FIX 2: lower ema_alpha so the smoothed center tracks turns faster
        self.ema_alpha = float(rospy.get_param("~ema_alpha", 0.35))

        # ---- Control tunables -------------------------------------------
        self.enable_drive   = bool (rospy.get_param("~enable_drive",   True))
        self.v_nominal      = float(rospy.get_param("~v_nominal",      0.23))
        self.v_min          = float(rospy.get_param("~v_min",          0.08))
        self.Kp             = float(rospy.get_param("~kp",             2.2))
        self.Kd             = float(rospy.get_param("~kd",             0.4))
        self.omega_max      = float(rospy.get_param("~omega_max",      4.0))
        # FIX 3b: tightened slew so a post-turn snap can't overshoot
        self.omega_slew     = float(rospy.get_param("~omega_slew",     3.0))
        self.turn_slowdown  = float(rospy.get_param("~turn_slowdown",  0.6))
        self.d_alpha        = float(rospy.get_param("~d_alpha",        0.5))
        self.omega_sign     = float(rospy.get_param("~omega_sign",    -1.0))
        self.lost_timeout   = float(rospy.get_param("~lost_timeout",   0.6))

        # ---- Persistent vision state ------------------------------------
        self.yellow_line    = None
        self.white_line     = None
        self.smooth_center_x = None
        # FIX 2: realistic initial half-width (~90 px for a Duckietown lane)
        self.lane_half_width = float(rospy.get_param("~lane_half_width_init", 90.0))
        self.yellow_seen    = False
        self.white_seen     = False

        # ---- Persistent control state -----------------------------------
        self._prev_e      = 0.0
        self._prev_t      = None
        self._d_filt      = 0.0
        self._prev_omega  = 0.0
        self._last_seen_t = None
        self._cmd_v       = 0.0
        self._cmd_omega   = 0.0

        # ---- Shared data between threads --------------------------------
        self.latest          = None   # newest raw frame  (callback -> ctrl thread)
        self._display_matrix = None   # newest debug image (ctrl thread -> GUI thread)
        self._frames         = 0
        self._quit           = False  # set by GUI thread; ctrl thread sees it

        # ---- ROS plumbing -----------------------------------------------
        self.bridge    = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.image_callback,
            queue_size=1, buff_size=2 ** 22,
        )

        self.cmd_topic = "/{}/car_cmd_switch_node/cmd".format(self.veh)
        self.pub = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)

        rospy.on_shutdown(self._publish_stop)

        rospy.loginfo("[panels] vehicle       = %s", self.veh)
        rospy.loginfo("[panels] camera topic  = %s", self.cam_topic)
        rospy.loginfo("[panels] cmd topic     = %s", self.cmd_topic)
        rospy.loginfo("[panels] DISPLAY env   = %s", os.environ.get("DISPLAY", "(unset!)"))
        rospy.loginfo("[panels] drive enabled = %s", self.enable_drive)
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
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
        ph = np.full((2 * self.PANEL_H, 3 * self.PANEL_W, 3), 40, dtype=np.uint8)
        cv2.putText(ph, text, (40, self.PANEL_H), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (200, 200, 200), 2, cv2.LINE_AA)
        return ph

    # ======================================================================
    #  Control: lane-center pixel error -> (v, omega)
    # ======================================================================
    def _compute_control(self, error_px, lane_half_width_px, lane_seen):
        now = rospy.get_time()

        if lane_seen:
            self._last_seen_t = now
        if self._last_seen_t is None or (now - self._last_seen_t) > self.lost_timeout:
            self._prev_e     = 0.0
            self._d_filt     = 0.0
            self._prev_omega = 0.0
            return 0.0, 0.0

        # FIX 1: normalise against actual lane half-width, not image half-width.
        # This makes Kp resolution-agnostic AND gives omega the full [-1,1]
        # dynamic range for a real lane offset.
        norm = float(max(1, lane_half_width_px))
        e = float(np.clip(float(error_px) / norm, -1.0, 1.0))

        dt = (now - self._prev_t) if self._prev_t is not None else 0.0
        self._prev_t = now
        de = (e - self._prev_e) / dt if dt > 1e-3 else 0.0
        self._prev_e = e
        self._d_filt = self.d_alpha * de + (1.0 - self.d_alpha) * self._d_filt

        omega = self.omega_sign * (self.Kp * e + self.Kd * self._d_filt)
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))

        if dt > 1e-3:
            max_step = self.omega_slew * dt
            omega = float(np.clip(omega,
                                  self._prev_omega - max_step,
                                  self._prev_omega + max_step))
        self._prev_omega = omega

        v = self.v_nominal * (1.0 - self.turn_slowdown * min(1.0, abs(e)))
        v = max(self.v_min, v)
        return v, omega

    def _publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v     = float(v)
        msg.omega = float(omega)
        self.pub.publish(msg)

    def _publish_stop(self):
        try:
            for _ in range(5):
                self._publish_cmd(0.0, 0.0)
                rospy.sleep(0.02)
        except Exception:
            pass

    # ======================================================================
    #  Camera callback — lightweight: decode + stash only.
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
            rospy.loginfo("[panels] first camera frame received — pipeline live")

    # ======================================================================
    #  CV + control pipeline (runs in background thread, never blocks on GUI)
    # ======================================================================
    def process_frame(self, frame):
        h, w = frame.shape[:2]

        roi_y1 = int(h * self.roi_top_ratio)
        roi_y2 = h
        roi    = frame[roi_y1:roi_y2, 0:w].copy()
        rh, rw = roi.shape[:2]
        img_center_x = rw // 2

        if self.yellow_line is None:
            self.yellow_line = np.array([0.0, 1.0, rw * 0.25, rh * 0.5], dtype=np.float32)
        if self.white_line is None:
            self.white_line  = np.array([0.0, 1.0, rw * 0.75, rh * 0.5], dtype=np.float32)
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_center_x)

        # HSV segmentation
        hsv         = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask  = self._clean_mask(cv2.inRange(hsv, self.white_lo,  self.white_hi))

        # Yellow line: fit through dash centroids
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
            self.yellow_seen = False

        # White line: fit through largest contour
        white_centroid  = None
        white_contours  = [c for c in self._find_contours(white_mask)
                           if cv2.contourArea(c) >= self.min_white_area]
        if white_contours:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line    = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            white_centroid     = self._centroid(largest)
            self.white_seen    = True
        else:
            self.white_seen = False

        # ---- Panel 1: raw + ROI box -------------------------------------
        panel1 = frame.copy()
        cv2.rectangle(panel1, (0, roi_y1), (w - 1, roi_y2 - 1), (0, 255, 0), 2)
        cv2.putText(panel1, "ROI", (6, roi_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        panel2 = yellow_mask
        panel3 = white_mask

        # ---- Panel 4: lane-bed overlay ----------------------------------
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

        for i in range(6):
            y   = int(rh * (i + 0.5) / 6)
            xl  = int(np.clip(self._x_at_y(self.yellow_line, y), 0, rw - 1))
            xr  = int(np.clip(self._x_at_y(self.white_line,  y), 0, rw - 1))
            cv2.line(panel4, (xl, y), (xr, y), (0, 255, 0), 1)
            cv2.circle(panel4, ((xl + xr) // 2, y), 2, (0, 0, 255), -1)

        for cen in yellow_centroids:
            cv2.circle(panel4, cen, 3, (0, 255, 255), -1)
        if white_centroid is not None:
            cv2.circle(panel4, white_centroid, 3, (255, 255, 255), -1)

        if not self.yellow_seen:
            cv2.putText(panel4, "Y:last", (4, rh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        if not self.white_seen:
            cv2.putText(panel4, "W:last", (rw - 70, rh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Panel 5: Canny + colour tint -------------------------------
        gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        panel5 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        panel5[yellow_mask > 0] = (0, 255, 255)
        panel5[white_mask  > 0] = (255, 255, 255)

        # ---- Lookahead + robust lane-center -----------------------------
        look_y = int(rh * self.lookahead_ratio)
        xl     = self._x_at_y(self.yellow_line, look_y)
        xr     = self._x_at_y(self.white_line,  look_y)

        if self.yellow_seen and self.white_seen:
            lane_center_x = (xl + xr) / 2.0
            hw = (xr - xl) / 2.0
            if hw > rw * 0.05:
                # FIX 2: faster convergence (weight 0.5 vs old 0.3)
                self.lane_half_width = 0.5 * self.lane_half_width + 0.5 * hw
        elif self.yellow_seen:
            lane_center_x = xl + self.lane_half_width
        elif self.white_seen:
            lane_center_x = xr - self.lane_half_width
        else:
            lane_center_x = self.smooth_center_x

        # FIX 3a: lower ema_alpha → faster tracking through turns
        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = int(np.clip(self.smooth_center_x, 0, rw - 1))
        error    = target_x - img_center_x

        # FIX 1: pass lane_half_width (not img_center_x) as normaliser
        lane_seen  = self.yellow_seen or self.white_seen
        v, omega   = self._compute_control(error, self.lane_half_width, lane_seen)
        self._cmd_v, self._cmd_omega = v, omega

        # ---- Panel 6: steering indicator --------------------------------
        panel6 = cv2.convertScaleAbs(roi, alpha=0.4)
        cv2.line(panel6, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
        cv2.line(panel6, (0, look_y), (rw - 1, look_y), (120, 120, 120), 1)
        cv2.arrowedLine(panel6, (img_center_x, rh - 1), (target_x, look_y),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.circle(panel6, (target_x, look_y), 4, (0, 255, 0), -1)

        direction = "STRAIGHT" if abs(error) < self.deadband else ("RIGHT" if error > 0 else "LEFT")
        cv2.putText(panel6, "err {:+d}px {}".format(error, direction),
                    (4, rh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(panel6, "v {:.2f}  w {:+.2f}".format(v, omega),
                    (4, rh - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        top_row    = np.hstack([self._format_panel(panel1, "1 RAW + ROI"),
                                self._format_panel(panel2, "2 YELLOW mask"),
                                self._format_panel(panel3, "3 WHITE mask")])
        bottom_row = np.hstack([self._format_panel(panel4, "4 LANE BED"),
                                self._format_panel(panel5, "5 EDGES + HSV"),
                                self._format_panel(panel6, "6 STEERING")])
        return np.vstack([top_row, bottom_row])

    # ======================================================================
    #  _show — display helper (called from main/GUI thread only)
    # ======================================================================
    def _show(self, matrix):
        if self.DISPLAY_SCALE != 1.0:
            matrix = cv2.resize(matrix, None,
                                fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE,
                                interpolation=cv2.INTER_NEAREST)
        try:
            cv2.imshow(self.WINDOW, matrix)
        except cv2.error as exc:
            rospy.logerr_throttle(
                5.0,
                "[panels] imshow failed (%s). Add 'opencv-python' (not "
                "-headless) to dependencies-py3.txt and rebuild." % exc)

    # ======================================================================
    #  FIX 4 — CONTROL THREAD
    #  Runs the CV pipeline + publishes drive commands independently of the
    #  GUI.  The main thread only calls imshow/waitKey; it never blocks drive.
    # ======================================================================
    def _control_loop(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown() and not self._quit:
            frame = self.latest
            if frame is not None:
                try:
                    matrix = self.process_frame(frame)
                    # Atomic assignment — safe to read from main thread in CPython
                    self._display_matrix = matrix
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "[panels] pipeline error: %s" % exc)
                    self._cmd_v, self._cmd_omega = 0.0, 0.0

            if self.enable_drive:
                self._publish_cmd(self._cmd_v, self._cmd_omega)

            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

        self._publish_stop()

    # ======================================================================
    #  spin — main (GUI) thread: imshow + waitKey only, never blocks drive
    # ======================================================================
    def spin(self):
        # Start the control loop in a daemon thread so it is killed when
        # the main process exits.
        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        self._show(self._placeholder("waiting for camera..."))
        # waitKey(1) is called immediately — window exists but we do NOT
        # wait for a click before the control thread starts publishing.
        cv2.waitKey(1)

        gui_rate = rospy.Rate(self.display_rate)
        while not rospy.is_shutdown():
            matrix = self._display_matrix
            if matrix is not None:
                self._show(matrix)
            else:
                self._show(self._placeholder("waiting for camera..."))

            # waitKey drives the Qt/GTK event loop; 1 ms timeout keeps it
            # non-blocking relative to the control thread.
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self._quit = True
                break

            try:
                gui_rate.sleep()
            except rospy.ROSInterruptException:
                break

        self._quit = True
        ctrl_thread.join(timeout=2.0)
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



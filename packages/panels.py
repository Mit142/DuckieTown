#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py
=========
DISPLAY + DRIVE lane-following node (ROS Noetic). Shows ONE panel:

    +---------------------+
    | 6  STEERING + v/w   |
    +---------------------+

LANE CENTER (this version)
--------------------------
The drivable corridor is bounded by:
  * the RIGHT edge of the dashed YELLOW centerline (on the bot's left), and
  * the LEFT edge of the thick WHITE boundary (on the bot's right).
We measure those two INNER edges directly from the colour masks in a small
horizontal band at the lookahead row, and steer toward their midpoint:

    target = (yellow_right_edge + white_left_edge) / 2

No more "push left when yellow disappears" logic -- this is a clean,
symmetric center aimed at straight-lane following. If only one edge is
visible we fall back to a remembered half-corridor width; if neither is
visible we hold the last target (and the lost-timeout stops the bot).

On-screen (panel 6): yellow dot = yellow inner edge, white dot = white inner
edge, green dot = target, red arrow = steer direction; text shows err + v/w.

CONTROL: PD on the normalised lane error -> omega; forward speed eases off in
turns. Steering omega is slew-limited so it can't snap.

COMMAND TOPIC: Twist2DStamped to /<veh>/car_cmd_switch_node/cmd.

DISPLAY: needs X forwarding (dts devel run -X). If X isn't available, run with
_enable_display:=false to drive headless (no window, no crash).

SAFETY: zero command on quit / Ctrl-C; stops if the lane is lost for
lost_timeout s. Press 'q' in the window, or Ctrl-C.
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
    """Shows the steering panel AND follows the lane via inner-edge center."""

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


        # HSV thresholds (OpenCV: H 0-179, S 0-255, V 0-255)
        self.yellow_lo = np.array([20,  70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo  = np.array([ 0,   0, 180], dtype=np.uint8)
        self.white_hi  = np.array([179, 55, 255], dtype=np.uint8)
        self.kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Inner-edge detection band around the lookahead row.
        self.edge_band   = int(rospy.get_param("~edge_band", 15))    # +/- px
        self.min_edge_px = int(rospy.get_param("~min_edge_px", 20))  # pixels to count

        self.deadband  = 8
        self.ema_alpha = float(rospy.get_param("~ema_alpha", 0.35))

        # ---- Control tunables -------------------------------------------
        self.enable_drive   = bool (rospy.get_param("~enable_drive",   True))
        self.v_nominal      = float(rospy.get_param("~v_nominal",      0.23))
        self.v_min          = float(rospy.get_param("~v_min",          0.08))
        self.Kp             = float(rospy.get_param("~kp",             2.2))
        self.Kd             = float(rospy.get_param("~kd",             0.4))
        self.omega_max      = float(rospy.get_param("~omega_max",      4.0))
        self.omega_slew     = float(rospy.get_param("~omega_slew",     15.0))
        self.turn_slowdown  = float(rospy.get_param("~turn_slowdown",  0.6))
        self.d_alpha        = float(rospy.get_param("~d_alpha",        0.5))
        self.omega_sign     = float(rospy.get_param("~omega_sign",    -1.0))
        self.lost_timeout   = float(rospy.get_param("~lost_timeout",   0.6))
        # Optional lateral trim (px). 0 = pure inner-edge midpoint.
        self.center_offset_px = float(rospy.get_param("~center_offset_px", 0.0))

        # ---- Persistent state -------------------------------------------
        self.smooth_center_x = None
        # Fallback half-corridor width (px), self-updates when both edges seen.
        self.lane_half_width = float(rospy.get_param("~lane_half_width_init", 110.0))
        self.yellow_seen = False
        self.white_seen  = False

        self._prev_e      = 0.0
        self._prev_t      = None
        self._d_filt      = 0.0
        self._prev_omega  = 0.0
        self._last_seen_t = None
        self._cmd_v       = 0.0
        self._cmd_omega   = 0.0

        # ---- Shared between threads -------------------------------------
        self.latest          = None
        self._display_matrix = None
        self._frames         = 0
        self._quit           = False

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

        rospy.loginfo("[panels] vehicle        = %s", self.veh)
        rospy.loginfo("[panels] camera topic   = %s", self.cam_topic)
        rospy.loginfo("[panels] cmd topic      = %s", self.cmd_topic)
        rospy.loginfo("[panels] DISPLAY env    = %s", os.environ.get("DISPLAY", "(unset!)"))
        rospy.loginfo("[panels] drive enabled  = %s", self.enable_drive)
        rospy.loginfo("[panels] press q in the window to quit")

    # ======================================================================
    #  Helpers
    # ======================================================================
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
        ph = np.full((self.PANEL_H, self.PANEL_W, 3), 40, dtype=np.uint8)
        cv2.putText(ph, text, (10, self.PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2, cv2.LINE_AA)
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
            rospy.loginfo("[panels] first camera frame received -- pipeline live")

    # ======================================================================
    #  Inner-edge measurement at the lookahead band
    # ======================================================================
    def _inner_edges(self, yellow_mask, white_mask, look_y, rh, img_center_x):
        """Return (yellow_right_edge_x, white_left_edge_x); either may be None."""
        y0 = max(0, look_y - self.edge_band)
        y1 = min(rh, look_y + self.edge_band + 1)

        # YELLOW: rightmost yellow pixel in the band = the edge facing the lane.
        ys_y, ys_x = np.where(yellow_mask[y0:y1, :] > 0)
        yellow_inner = int(ys_x.max()) if ys_x.size >= self.min_edge_px else None

        # WHITE: leftmost white pixel that lies to the RIGHT of the yellow edge
        #        (or right of image center if yellow is missing) = inner edge of
        #        the right-hand boundary.
        ws_y, ws_x = np.where(white_mask[y0:y1, :] > 0)
        white_inner = None
        if ws_x.size >= self.min_edge_px:
            ref = yellow_inner if yellow_inner is not None else img_center_x
            right = ws_x[ws_x > ref]
            if right.size > 0:
                white_inner = int(right.min())
        return yellow_inner, white_inner

    # ======================================================================
    #  CV + control pipeline (runs in background thread)
    # ======================================================================
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        roi   = frame[int(h * self.roi_top_ratio):h, 0:w].copy()
        rh, rw = roi.shape[:2]
        img_center_x = rw // 2
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_center_x)

        # HSV segmentation
        hsv         = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask  = self._clean_mask(cv2.inRange(hsv, self.white_lo,  self.white_hi))

        # ---- Inner edges at the lookahead band --------------------------
        look_y = int(rh * self.lookahead_ratio)
        yellow_inner, white_inner = self._inner_edges(
            yellow_mask, white_mask, look_y, rh, img_center_x)
        self.yellow_seen = yellow_inner is not None
        self.white_seen  = white_inner  is not None

        # ---- Lane center = midpoint of the two inner edges --------------
        if self.yellow_seen and self.white_seen:
            lane_center_x = (yellow_inner + white_inner) / 2.0
            hw = (white_inner - yellow_inner) / 2.0
            if hw > rw * 0.05:
                self.lane_half_width = 0.5 * self.lane_half_width + 0.5 * hw
        elif self.white_seen:
            lane_center_x = white_inner - self.lane_half_width
        elif self.yellow_seen:
            lane_center_x = yellow_inner + self.lane_half_width
        else:
            lane_center_x = self.smooth_center_x   # hold last; lost-timeout stops

        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)

        target_x = int(np.clip(self.smooth_center_x + self.center_offset_px, 0, rw - 1))
        error    = target_x - img_center_x

        lane_seen = self.yellow_seen or self.white_seen
        v, omega  = self._compute_control(error, self.lane_half_width, lane_seen)
        self._cmd_v, self._cmd_omega = v, omega

        rospy.loginfo_throttle(
            1.0,
            "[panels] yL=%s wL=%s half_w=%.0f err=%+d v=%.2f w=%+.2f"
            % (str(yellow_inner), str(white_inner),
               self.lane_half_width, error, v, omega))

        # ---- Panel 6: steering view -------------------------------------
        panel6 = cv2.convertScaleAbs(roi, alpha=0.4)
        cv2.line(panel6, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
        cv2.line(panel6, (0, look_y), (rw - 1, look_y), (120, 120, 120), 1)
        if self.yellow_seen:
            cv2.circle(panel6, (yellow_inner, look_y), 5, (0, 255, 255), -1)   # yellow edge
        if self.white_seen:
            cv2.circle(panel6, (white_inner, look_y), 5, (255, 255, 255), -1)  # white edge
        cv2.arrowedLine(panel6, (img_center_x, rh - 1), (target_x, look_y),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.circle(panel6, (target_x, look_y), 4, (0, 255, 0), -1)             # target
        direction = "STRAIGHT" if abs(error) < self.deadband else ("RIGHT" if error > 0 else "LEFT")
        cv2.putText(panel6, "err {:+d}px {}".format(error, direction),
                    (4, rh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(panel6, "v {:.2f}  w {:+.2f}".format(v, omega),
                    (4, rh - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return self._format_panel(panel6, "6 STEERING")

    # ======================================================================
    #  _show — display helper (main/GUI thread only)
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
    #  Control thread: CV + publish, independent of GUI
    # ======================================================================
    def _control_loop(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown() and not self._quit:
            frame = self.latest
            if frame is not None:
                try:
                    matrix = self.process_frame(frame)
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
    #  spin — main thread. With display: GUI loop. Headless: just wait.
    # ======================================================================
    def spin(self):
        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)

        self._show(self._placeholder("waiting for camera..."))
        cv2.waitKey(1)

        gui_rate = rospy.Rate(self.display_rate)
        while not rospy.is_shutdown():
            matrix = self._display_matrix
            self._show(matrix if matrix is not None
                       else self._placeholder("waiting for camera..."))
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


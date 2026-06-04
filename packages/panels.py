#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py
=========
DISPLAY variant of the 6-panel lane-debug node (ROS Noetic).

Same computer-vision pipeline as the publishing lane_debug_node, but instead
of publishing a CompressedImage it shows the 2x3 matrix directly in an OpenCV
window via cv2.imshow(). Runs on the bot; the window is forwarded to your
laptop by `dts devel run -X` (XLaunch / VcXsrv on WSL2). Press 'q' to quit.

    +---------------------+---------------------+---------------------+
    | 1  RAW + ROI box    | 2  YELLOW mask      | 3  WHITE mask       |
    +---------------------+---------------------+---------------------+
    | 4  LANE BED overlay | 5  EDGES / combined | 6  STEERING vector  |
    +---------------------+---------------------+---------------------+

Real-time notes:
  - The callback ONLY decodes + stores the newest frame (cheap). All the heavy
    CV work runs in the main loop on the latest frame, so stale frames are
    dropped instead of queuing -> the panels track live motion.
  - DISPLAY_SCALE shrinks ONLY the final window. The panels stay full-size for
    the math; lower it if the X-forwarded window lags over the network.
  - This needs GUI OpenCV (opencv-python, NOT -headless) for imshow -- the same
    build your camera_view.py used. Do not revert that Dockerfile change here.

Run on the bot:
    rosrun <your_package> panels.py
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage


class LanePanelsView:
    """Builds and SHOWS the 6-panel lane-detection debug matrix."""

    WINDOW = "lane panels"
    DISPLAY_SCALE = 1.0   # 1.0 = full 960x480 matrix; lower if the window lags

    def __init__(self):
        rospy.init_node("panels", anonymous=False)

        # Vehicle name -> namespaced camera topic.
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # Geometry. WORK_* = working resolution; PANEL_* = each panel.
        self.WORK_W, self.WORK_H = 640, 480
        self.PANEL_W, self.PANEL_H = 320, 240

        # Tunables (overridable via the ROS param server).
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.35))
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
        self.deadband = 8
        self.ema_alpha = 0.4

        # Persistent state for graceful fallback / smoothing.
        # cv2.fitLine returns [vx, vy, x0, y0]: unit dir (vx,vy) + point (x0,y0).
        self.yellow_line = None
        self.white_line = None
        self.smooth_center_x = None
        self.yellow_seen = False
        self.white_seen = False

        # Newest raw frame (set in callback, consumed in main loop).
        self.latest = None

        # ROS plumbing -- subscriber only, no publisher, no GUI here.
        self.bridge = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        self.sub = rospy.Subscriber(
            self.cam_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 22,   # small buffer -> ROS drops stale frames
        )

        rospy.loginfo("[panels] vehicle = %s", self.veh)
        rospy.loginfo("[panels] subscribed to %s", self.cam_topic)
        rospy.loginfo("[panels] display scale = %.2f", self.DISPLAY_SCALE)
        rospy.loginfo("[panels] press q in the window to quit")

    # ======================================================================
    #  Helpers
    # ======================================================================
    @staticmethod
    def _find_contours(mask):
        """Contours, compatible with OpenCV 3 and 4 ([-2] grabs the list)."""
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    @staticmethod
    def _centroid(contour):
        """Centroid (cx, cy) via image moments, or None if degenerate."""
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

    @staticmethod
    def _x_at_y(line, y):
        """Project a fitted line to row `y`: x = x0 + (y - y0) * (vx / vy)."""
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

    # ======================================================================
    #  Pipeline -> assembled 2x3 BGR matrix
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

        # ---------- PANEL 6: steering indicator ----------------------------
        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line, look_y)
        lane_center_x = (xl + xr) / 2.0

        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = int(np.clip(self.smooth_center_x, 0, rw - 1))
        error = target_x - img_center_x

        panel6 = cv2.convertScaleAbs(roi, alpha=0.4)
        bot_origin = (img_center_x, rh - 1)
        cv2.line(panel6, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
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
                    (4, rh - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
    #  Main loop: process newest frame -> show (GUI on main thread).
    # ======================================================================
    def spin(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        rate = rospy.Rate(self.display_rate)
        while not rospy.is_shutdown():
            frame = self.latest
            if frame is not None:
                try:
                    matrix = self.process_frame(frame)
                except Exception as exc:  # stay alive on a bad frame
                    rospy.logwarn_throttle(2.0, "[panels] pipeline error: %s" % exc)
                    matrix = None

                if matrix is not None:
                    if self.DISPLAY_SCALE != 1.0:
                        matrix = cv2.resize(
                            matrix, None,
                            fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE,
                            interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(self.WINDOW, matrix)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        cv2.destroyAllWindows()


def main():
    node = LanePanelsView()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



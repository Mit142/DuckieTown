#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_debug_viewer.py
====================
Broadcasting variant of the Duckiebot lane-debug node (ROS Noetic).

Identical computer-vision pipeline to the original layout, but instead of
displaying directly to an X server via cv2.imshow(), it packages the 2x3 matrix
and streams it as a live ROS compressed image topic.

    +---------------------+---------------------+---------------------+
    | 1  RAW + ROI box    | 2  YELLOW mask      | 3  WHITE mask       |
    +---------------------+---------------------+---------------------+
    | 4  LANE BED overlay | 5  EDGES / combined | 6  STEERING vector  |
    +---------------------+---------------------+---------------------+

View this on your laptop by running:
    dts gui entebot208
And select the topic `/{vehicle_name}/lane_debug/compressed`.
"""

import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage


class LaneDebugViewer:
    """Builds and publishes the 6-panel lane-detection debug matrix over ROS."""

    def __init__(self):
        rospy.init_node("lane_debug_viewer", anonymous=False)

        # Vehicle configuration
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # Geometry constants
        self.WORK_W, self.WORK_H = 640, 480
        self.PANEL_W, self.PANEL_H = 320, 240

        # Tunable runtime parameters
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.35))
        self.display_rate = float(rospy.get_param("~display_rate", 20.0))  # Hz

        # HSV colour thresholds (OpenCV ranges: H 0-179, S 0-255, V 0-255)
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

        # Persistent state for smoothing
        self.yellow_line = None
        self.white_line = None
        self.smooth_center_x = None
        self.yellow_seen = False
        self.white_seen = False
        self.latest_matrix = None

        # ROS plumbing
        self.bridge = CvBridge()
        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        self.debug_topic = "/{}/lane_debug/compressed".format(self.veh)

        # Subscriber
        self.sub = rospy.Subscriber(
            self.cam_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        # Publisher for streaming the visualization matrix
        self.pub_debug = rospy.Publisher(self.debug_topic, CompressedImage, queue_size=1)

        rospy.loginfo("[lane_debug_viewer] vehicle = %s", self.veh)
        rospy.loginfo("[lane_debug_viewer] subscribed to %s", self.cam_topic)
        rospy.loginfo("[lane_debug_viewer] publishing stream to %s", self.debug_topic)

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

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[lane_debug_viewer] CvBridge decode failed: %s", exc)
            return

        frame = cv2.resize(frame, (self.WORK_W, self.WORK_H))

        try:
            self.latest_matrix = self.process_frame(frame)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[lane_debug_viewer] pipeline error: %s" % exc)

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        # ---------- Region of Interest ----------------
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

        # ---------- Colour segmentation in HSV -----------------------------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask = self._clean_mask(cv2.inRange(hsv, self.white_lo, self.white_hi))

        # ---------- Fit the LEFT (yellow, dashed) lane ---------------------
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

        # ---------- Fit the RIGHT (white, solid) lane ----------------------
        white_centroid = None
        white_contours = [c for c in self._find_contours(white_mask)
                          if cv2.contourArea(c) >= self.min_white_area]
        if white_contours:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            white_centroid = self._centroid(largest)
            self.white_seen = True
        else:
            self.white_seen = False

        # PANEL 1 - raw image with the ROI box
        panel1 = frame.copy()
        cv2.rectangle(panel1, (0, roi_y1), (w - 1, roi_y2 - 1), (0, 255, 0), 2)
        cv2.putText(panel1, "ROI", (6, roi_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # PANELS 2 & 3 - masks
        panel2 = yellow_mask
        panel3 = white_mask

        # PANEL 4 - the "lane bed" overlay
        panel4 = roi.copy()
        y_top, y_bot = 0, rh - 1
        cv2.line(panel4,
                 (self._x_at_y(self

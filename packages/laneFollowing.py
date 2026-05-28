#!/usr/bin/env python3
"""
Lane Following Node for Duckiebot
==================================
Uses only standard ROS messages (no duckietown_msgs dependency).
Publishes wheel commands directly to wheels_driver_node.

Road layout:
  [white line] --- [yellow dashed center] --- [white line]
                        ^ stay here (right side)
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64


class LaneFollowingNode:
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo(f"[{self.node_name}] Initializing...")

        # --- Parameters ---
        self.v_bar     = rospy.get_param("~v_bar",     0.3)   # forward speed (0.0 - 1.0 for wheels)
        self.k_theta   = rospy.get_param("~k_theta",   2.0)   # heading gain
        self.k_d       = rospy.get_param("~k_d",       3.0)   # lateral offset gain
        self.d_offset  = rospy.get_param("~d_offset", -0.10)  # desired offset, negative = right of center
        self.omega_max = rospy.get_param("~omega_max",  5.0)  # max angular correction
        self.baseline  = 0.1                                   # Duckiebot wheel baseline (m)

        # --- CV Bridge ---
        self.bridge = CvBridge()
        self.active = True

        # --- Publishers (direct to wheel driver) ---
        veh = rospy.get_param("~veh", "entebot208")
        self.pub_left  = rospy.Publisher(
            f"/{veh}/wheels_driver_node/wheels_cmd_left",  Float64, queue_size=1
        )
        self.pub_right = rospy.Publisher(
            f"/{veh}/wheels_driver_node/wheels_cmd_right", Float64, queue_size=1
        )

        # --- Subscriber: raw camera image ---
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage,
            self.cb_image, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(f"[{self.node_name}] Ready — waiting for images.")

    def cb_image(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"[{self.node_name}] Image decode error: {e}")
            return

        d, phi = self._detect_lane(img)

        if d is None:
            rospy.logwarn_throttle(2.0, f"[{self.node_name}] No lane detected — going straight slowly.")
            self._send_wheels(self.v_bar * 0.3, 0.0)
            return

        self._send_wheels(self.v_bar, self.k_d * d + self.k_theta * phi)

    def _detect_lane(self, img):
        h, w = img.shape[:2]
        roi  = img[int(h * 0.55):, :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(
            hsv, np.array([0, 0, 150]), np.array([180, 60, 255])
        )
        yellow_mask = cv2.inRange(
            hsv, np.array([20, 100, 100]), np.array([35, 255, 255])
        )

        kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask  = cv2.morphologyEx(white_mask,  cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        roi_h, roi_w = roi.shape[:2]
        cx_white     = self._centroid_x(white_mask)
        cx_yellow    = self._centroid_x(yellow_mask)

        if cx_white is None and cx_yellow is None:
            return None, None

        px_per_m = roi_w / 0.35

        if cx_yellow is not None and cx_white is not None:
            lane_center_px = (cx_yellow + cx_white) / 2.0
        elif cx_yellow is not None:
            lane_center_px = cx_yellow + roi_w * 0.25
        else:
            lane_center_px = cx_white - roi_w * 0.25

        img_center_px = roi_w / 2.0
        d   = (img_center_px - lane_center_px) / px_per_m - self.d_offset
        phi = self._estimate_heading(yellow_mask, white_mask)

        return d, phi

    def _centroid_x(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 200:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return M["m10"] / M["m00"]

    def _estimate_heading(self, yellow_mask, white_mask):
        combined = cv2.bitwise_or(yellow_mask, white_mask)
        edges    = cv2.Canny(combined, 50, 150)
        lines    = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                   threshold=30, minLineLength=20, maxLineGap=10)
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1)
                if abs(angle) > np.pi / 6:
                    angles.append(angle)

        if not angles:
            return 0.0

        median_angle = np.median(angles)
        phi = -(median_angle - np.sign(median_angle) * np.pi / 2.0)
        return float(np.clip(phi, -np.pi / 4, np.pi / 4))

    def _send_wheels(self, v, omega):
        if not self.active:
            self._send_stop()
            return

        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        left  = float(np.clip(v - 0.5 * omega * self.baseline, -1.0, 1.0))
        right = float(np.clip(v + 0.5 * omega * self.baseline, -1.0, 1.0))

        self.pub_left.publish(Float64(left))
        self.pub_right.publish(Float64(right))

    def _send_stop(self):
        self.pub_left.publish(Float64(0.0))
        self.pub_right.publish(Float64(0.0))

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down — stopping wheels.")
        self.active = False
        self._send_stop()


if __name__ == "__main__":
    rospy.init_node("lane_following_node", anonymous=False)
    node = LaneFollowingNode()
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()

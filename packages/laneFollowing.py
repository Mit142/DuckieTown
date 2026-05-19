#!/usr/bin/env python3
"""
Lane Following Node for Duckiebot
==================================
Follows the right side of a yellow dashed centerline,
staying within white lane boundaries.

Road layout:
  [white line] --- [yellow dashed center] --- [white line]
                        ^ stay here (right side)

Topics subscribed:
  ~image/compressed        : camera image
  ~lane_pose               : (optional) if using upstream lane filter

Topics published:
  ~car_cmd                 : wheel velocity commands (Twist2DStamped)

Tune these parameters via rosparam or the Duckietown dashboard:
  ~v_bar      : nominal forward speed (m/s)
  ~k_theta    : heading error gain
  ~k_d        : lateral offset gain
  ~d_offset   : desired lateral offset from centerline (negative = right side)
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import Twist2DStamped, LanePose


class LaneFollowingNode:
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo(f"[{self.node_name}] Initializing lane following node...")

        # --- Parameters ---
        self.v_bar     = rospy.get_param("~v_bar",     0.2)    # forward speed m/s
        self.k_theta   = rospy.get_param("~k_theta",   2.0)    # heading gain
        self.k_d       = rospy.get_param("~k_d",       3.0)    # lateral offset gain
        self.d_offset  = rospy.get_param("~d_offset", -0.10)   # desired offset (m), negative = right of center
        self.omega_max = rospy.get_param("~omega_max",  8.0)    # max angular velocity rad/s

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- State ---
        self.last_d     = 0.0   # lateral offset from centerline
        self.last_phi   = 0.0   # heading error (rad)
        self.active     = True

        # --- Publishers ---
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd", Twist2DStamped, queue_size=1
        )

        # --- Subscribers ---
        # Option A: use the upstream lane filter pose directly (preferred)
        self.sub_lane_pose = rospy.Subscriber(
            "~lane_pose", LanePose, self.cb_lane_pose, queue_size=1
        )

        # Option B: raw image fallback (used if lane_pose not available)
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.cb_image, queue_size=1,
            buff_size=2**24
        )

        self.use_raw_image = False  # flipped to True if no lane_pose arrives
        self._lane_pose_received = False
        self._check_timer = rospy.Timer(
            rospy.Duration(3.0), self._check_lane_pose_source, oneshot=True
        )

        rospy.loginfo(f"[{self.node_name}] Ready.")

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------

    def _check_lane_pose_source(self, _event):
        if not self._lane_pose_received:
            rospy.logwarn(
                f"[{self.node_name}] No lane_pose received — falling back to raw image processing."
            )
            self.use_raw_image = True

    # ------------------------------------------------------------------
    # Callback: upstream LanePose (from dt-core lane filter)
    # ------------------------------------------------------------------

    def cb_lane_pose(self, msg):
        self._lane_pose_received = True
        self.use_raw_image = False
        self.last_d   = msg.d      # lateral offset (m)
        self.last_phi = msg.phi    # heading error (rad)
        self._publish_command(self.last_d, self.last_phi)

    # ------------------------------------------------------------------
    # Callback: raw compressed image (fallback)
    # ------------------------------------------------------------------

    def cb_image(self, msg):
        if not self.use_raw_image:
            return
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"[{self.node_name}] Image decode error: {e}")
            return

        d, phi = self._detect_lane(img)
        if d is None:
            # No lines found — slow down and go straight
            self._publish_command(0.0, 0.0, speed_scale=0.3)
            return

        self.last_d   = d
        self.last_phi = phi
        self._publish_command(d, phi)

    # ------------------------------------------------------------------
    # Lane detection from raw image
    # ------------------------------------------------------------------

    def _detect_lane(self, img):
        """
        Returns (d, phi):
          d   – lateral offset from desired position (m), positive = left of target
          phi – heading error (rad), positive = pointing left
        Returns (None, None) if detection fails.
        """
        h, w = img.shape[:2]

        # Only look at the bottom third of the image (road ahead)
        roi = img[int(h * 0.55):, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- White lane boundaries ---
        white_mask = cv2.inRange(
            hsv,
            np.array([0,   0, 150]),
            np.array([180, 60, 255])
        )

        # --- Yellow center line ---
        yellow_mask = cv2.inRange(
            hsv,
            np.array([20, 100, 100]),
            np.array([35, 255, 255])
        )

        # Clean up masks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask  = cv2.morphologyEx(white_mask,  cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        roi_h, roi_w = roi.shape[:2]
        cx_white  = self._mask_centroid_x(white_mask)
        cx_yellow = self._mask_centroid_x(yellow_mask)

        if cx_white is None and cx_yellow is None:
            return None, None

        # Estimate lane center and heading from detected lines
        # Duckietown standard lane width ≈ 0.23 m, image width maps to ~0.35 m in ROI
        px_per_m = roi_w / 0.35  # rough calibration

        # Desired x position: right of yellow line (or left of white right boundary)
        if cx_yellow is not None and cx_white is not None:
            # Both lines visible — ideal case
            lane_center_px = (cx_yellow + cx_white) / 2.0
            lane_width_px  = abs(cx_white - cx_yellow)
        elif cx_yellow is not None:
            # Only yellow visible — estimate lane center
            lane_center_px = cx_yellow + roi_w * 0.25
            lane_width_px  = roi_w * 0.5
        else:
            # Only white right boundary visible
            lane_center_px = cx_white - roi_w * 0.25
            lane_width_px  = roi_w * 0.5

        # Image center = where bot is heading
        img_center_px = roi_w / 2.0

        # d: lateral offset in meters (positive = bot is left of lane center)
        d_raw = (img_center_px - lane_center_px) / px_per_m

        # Apply desired offset (we want to be slightly right of center)
        d = d_raw - self.d_offset

        # phi: estimate heading from line angle using Hough lines
        phi = self._estimate_heading(yellow_mask, white_mask, roi_w)

        return d, phi

    def _mask_centroid_x(self, mask):
        """Return x-centroid of largest contour in mask, or None."""
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

    def _estimate_heading(self, yellow_mask, white_mask, roi_w):
        """Estimate heading error phi from Hough lines on the combined mask."""
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
                # Only keep near-vertical lines (lane lines should be ~90 deg)
                if abs(angle) > np.pi / 6:
                    angles.append(angle)

        if not angles:
            return 0.0

        median_angle = np.median(angles)
        # Convert: vertical lane lines appear at ~±pi/2; deviation from pi/2 is heading error
        phi = -(median_angle - np.sign(median_angle) * np.pi / 2.0)
        return float(np.clip(phi, -np.pi / 4, np.pi / 4))

    # ------------------------------------------------------------------
    # Control law & publisher
    # ------------------------------------------------------------------

    def _publish_command(self, d, phi, speed_scale=1.0):
        """
        Simple P controller:
          omega = k_d * d + k_theta * phi
        """
        if not self.active:
            self._send_stop()
            return

        omega = self.k_d * d + self.k_theta * phi
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        v     = self.v_bar * speed_scale

        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v     = v
        cmd.omega = omega

        self.pub_car_cmd.publish(cmd)

    def _send_stop(self):
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v     = 0.0
        cmd.omega = 0.0
        self.pub_car_cmd.publish(cmd)

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down — sending stop command.")
        self.active = False
        self._send_stop()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rospy.init_node("lane_following_node", anonymous=False)
    node = LaneFollowingNode()
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()

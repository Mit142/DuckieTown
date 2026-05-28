#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped

class LaneFollowerNode:
    def __init__(self):
        rospy.init_node('custom_lane_follower', anonymous=True)
       
        # Change 'duckiebot_name' to your robot's actual hostname
        self.veh = rospy.get_param("~veh", "duckiebot_name")
       
        # Publishers and Subscribers
        self.sub_cam = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.camera_callback, queue_size=1)
        self.pub_cmd = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
       
        # Control Variables (TUNE THESE)
        self.v_base = 0.25      # Base forward speed (m/s)
        self.Kp = 0.005         # Proportional gain for steering
        self.Kd = 0.002         # Derivative gain for steering
        self.last_error = 0.0
       
        # Target image center (will update when first image arrives)
        self.image_center_x = 320

        rospy.loginfo("Lane Follower Node Initialized.")

    def camera_callback(self, msg):
        # 1. Convert CompressedImage to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
       
        height, width, _ = frame.shape
        self.image_center_x = width // 2
       
        # 2. Crop Image (Region of Interest) - Only look at the bottom half of the screen
        roi = frame[height//2:height, :]
       
        # 3. Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
       
        # 4. Define Color Ranges (TUNE THESE based on your room's lighting)
        # Yellow threshold (Left Lane)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
       
        # White threshold (Right Lane)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
       
        # 5. Find Centroids of the lines
        cx_yellow, cy_yellow = self.get_centroid(mask_yellow)
        cx_white, cy_white = self.get_centroid(mask_white)
       
        # 6. Calculate the Target Point and Error
        error = 0.0
       
        # If we see both lines, aim for the exact middle
        if cx_yellow is not None and cx_white is not None:
            target_x = (cx_yellow + cx_white) // 2
            error = self.image_center_x - target_x
           
        # If we only see the yellow line (left), offset to the right
        elif cx_yellow is not None:
            target_x = cx_yellow + 150 # 150 pixels to the right of the yellow line
            error = self.image_center_x - target_x
           
        # If we only see the white line (right), offset to the left
        elif cx_white is not None:
            target_x = cx_white - 150 # 150 pixels to the left of the white line
            error = self.image_center_x - target_x
           
        # 7. Compute PD Control for Steering
        omega = (self.Kp * error) + (self.Kd * (error - self.last_error))
        self.last_error = error
       
        # Slow down slightly if the turn is sharp (high error)
        current_v = self.v_base
        if abs(error) > 50:
            current_v = self.v_base * 0.8
           
        # 8. Publish Motor Commands
        self.publish_cmd(current_v, omega)

    def get_centroid(self, mask):
        # Calculate image moments to find the center of mass of the pixels
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return None, None

    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v          # Forward velocity
        msg.omega = omega  # Angular velocity (steering)
        self.pub_cmd.publish(msg)

if __name__ == '__main__':
    try:
        node = LaneFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

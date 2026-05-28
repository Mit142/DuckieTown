#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist  # Standard ROS velocity message

class LaneFollowerNode:
    def __init__(self):
        rospy.init_node('standard_lane_follower', anonymous=True)
       
        # Change 'duckiebot_name' to your robot's actual hostname
        self.veh = rospy.get_param("~veh", "duckiebot_name")
       
        # Subscribers and Publishers
        self.sub_cam = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.camera_callback, queue_size=1)
       
        # NOTE: Duckietown's default motor driver usually expects Twist2DStamped.
        # If you use standard Twist, you typically publish to /cmd_vel.
        # You may need to check your rostopic list to see if your bot listens to /cmd_vel!
        self.pub_cmd = rospy.Publisher(f"/{self.veh}/cmd_vel", Twist, queue_size=1)
       
        # Control Variables
        self.v_base = 0.25      # Base forward speed (m/s)
        self.Kp = 0.005         # Proportional gain for steering
        self.Kd = 0.002         # Derivative gain for steering
        self.last_error = 0.0
       
        self.image_center_x = 320

        rospy.loginfo("Standard Twist Lane Follower Node Initialized.")

    def camera_callback(self, msg):
        # 1. Convert CompressedImage to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
       
        height, width, _ = frame.shape
        self.image_center_x = width // 2
       
        # 2. Crop Image (Region of Interest)
        roi = frame[height//2:height, :]
       
        # 3. Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
       
        # 4. Define Color Ranges
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
       
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
       
        # 5. Find Centroids
        cx_yellow, cy_yellow = self.get_centroid(mask_yellow)
        cx_white, cy_white = self.get_centroid(mask_white)
       
        # 6. Calculate the Target Point and Error
        error = 0.0
       
        if cx_yellow is not None and cx_white is not None:
            target_x = (cx_yellow + cx_white) // 2
            error = self.image_center_x - target_x
           
        elif cx_yellow is not None:
            target_x = cx_yellow + 150
            error = self.image_center_x - target_x
           
        elif cx_white is not None:
            target_x = cx_white - 150
            error = self.image_center_x - target_x
           
        # 7. Compute PD Control
        omega = (self.Kp * error) + (self.Kd * (error - self.last_error))
        self.last_error = error
       
        current_v = self.v_base
        if abs(error) > 50:
            current_v = self.v_base * 0.8
           
        # 8. Publish Motor Commands
        self.publish_cmd(current_v, omega)

    def get_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return None, None

    def publish_cmd(self, v, omega):
        # Using standard geometry_msgs/Twist
        msg = Twist()
        msg.linear.x = v          # Drive forward
        msg.angular.z = omega     # Rotate around the Z axis (steering)
        self.pub_cmd.publish(msg)

if __name__ == '__main__':
    try:
        node = LaneFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


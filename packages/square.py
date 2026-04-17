#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Get the vehicle name from the environment variable (reliable fallback)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot')
        
        # Target the wheels driver directly to bypass kinematics and FSM blocks
        topic_name = f"/{self.veh}/wheels_driver_node/wheels_cmd"
        
        self.pub_cmd = rospy.Publisher(topic_name, WheelsCmdStamped, queue_size=1)

    def send_cmd(self, vel_left, vel_right, duration):
        """Publishes raw wheel commands for a set duration."""
        msg = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        
        rate = rospy.Rate(10) # 10 Hz publishing rate
        start_time = rospy.get_time()
        
        # Keep publishing until the duration is reached or ROS shuts down
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            self.pub_cmd.publish(msg)
            rate.sleep()
            
    def stop(self):
        """Halts the robot briefly to prevent drift before the next move."""
        self.send_cmd(vel_left=0.0, vel_right=0.0, duration=0.5)

    def execute_square(self):
        # Wait a moment for publishers to establish a connection
        rospy.sleep(1.0) 
        
        rospy.loginfo("Starting square trajectory...")
        for i in range(4):
            # 1. Drive forward (both wheels moving forward at 50% speed)
            rospy.loginfo(f"Side {i+1}: Driving straight...")
            self.send_cmd(vel_left=0.5, vel_right=0.5, duration=2.0)
            self.stop()
            
            # 2. Turn 90 degrees (left wheel forward, right wheel backward)
            rospy.loginfo(f"Side {i+1}: Turning 90 degrees...")
            # NOTE: You will likely need to tune the duration below (0.8) 
            # to achieve exactly a 90-degree turn based on your bot's battery/motors.
            self.send_cmd(vel_left=0.5, vel_right=-0.5, duration=0.8) 
            self.stop()
            
        rospy.loginfo("Square completed.")
        
    def on_shutdown(self):
        """Safety catch: Ensure wheels stop if the node is forcibly shut down."""
        rospy.loginfo("Shutting down... stopping motors.")
        self.stop()

if __name__ == '__main__':
    node = SquareDriverNode(node_name="square_driver_node")
    try:
        node.execute_square()
    except rospy.ROSInterruptException:
        pass


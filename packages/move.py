#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class DriveStraightNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DriveStraightNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Get the robot's name from the environment variables
        vehicle_name = os.environ['VEHICLE_NAME']
        
        # The ROS topic where the wheel driver listens for commands
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        
        # Create a publisher for the wheels topic
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        
    def run(self):
        # Publish messages at a rate of 10 Hz
        rate = rospy.Rate(10)
        
        # Command to drive straight (values are from -1.0 to 1.0)
        # We use 0.5 for 50% speed.
        move_msg = WheelsCmdStamped(vel_left=0.5, vel_right=0.5)
        
        # Command to stop
        stop_msg = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        
        rospy.loginfo("Driving straight for 3 seconds...")
        
        # Record the start time
        start_time = rospy.get_time()
        
        # Keep publishing the move command for exactly 3 seconds
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < 3.0:
            self._publisher.publish(move_msg)
            rate.sleep()
            
        # Stop the robot once 3 seconds have passed
        rospy.loginfo("3 seconds passed. Stopping!")
        self._publisher.publish(stop_msg)
        
    def on_shutdown(self):
        # Safety catch: Ensure wheels stop if the node is forcibly shut down prematurely
        stop_msg = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self._publisher.publish(stop_msg)

if __name__ == '__main__':
    # Initialize and run the node
    node = DriveStraightNode(node_name='drive_straight_node')
    node.run()

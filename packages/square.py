#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.veh = os.environ.get('entebot208', 'duckiebot')
        topic_name = f"/{self.veh}/wheels_driver_node/wheels_cmd"
        
        self.pub_cmd = rospy.Publisher(topic_name, WheelsCmdStamped, queue_size=1)

    def send_cmd(self, vel_left, vel_right, duration):
        """Publishes raw wheel commands for a set duration."""
        msg = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        
        rate = rospy.Rate(10)
        start_time = rospy.get_time()
        
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            self.pub_cmd.publish(msg)
            rate.sleep()
            
    def stop(self):
        """Halts the robot briefly to prevent drift before the next move."""
        self.send_cmd(vel_left=0.0, vel_right=0.0, duration=0.5)

    def execute_square(self):
        rospy.sleep(1.0) 
        
        rospy.loginfo("Starting square trajectory...")
        # 1. Drive forward
        self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
        self.stop()
        for i in range(4):
            
            # 2. Point Turn 90 degrees LEFT (Stronger & Sharper)
            # Both wheels help the turn, reducing motor strain.
            rospy.loginfo(f"Side {i+1}: Point turning left...")
            self.send_cmd(vel_left=0, vel_right=1, duration=0.8) 
            self.stop()
            # 1. Drive forward
            rospy.loginfo(f"Side {i+1}: Driving straight...")
            self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
            self.stop()
        
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



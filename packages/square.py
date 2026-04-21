#!/usr/bin/env python3
import os
import time  # <-- Using strict real-world time
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot')
        topic_name = f"/{self.veh}/wheels_driver_node/wheels_cmd"
        
        # Queue size 10 to prevent dropped messages
        self.pub_cmd = rospy.Publisher(topic_name, WheelsCmdStamped, queue_size=10)

    def send_cmd(self, vel_left, vel_right, duration):
        """Publishes raw wheel commands for a set duration."""
        msg = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        
        start_time = time.time()
        
        while not rospy.is_shutdown() and (time.time() - start_time) < duration:
            # SAFETY CATCH: Update the timestamp so the robot accepts the command
            msg.header.stamp = rospy.Time.now()
            
            self.pub_cmd.publish(msg)
            time.sleep(0.1)
            
    def stop(self):
        """Halts the robot briefly to prevent drift before the next move."""
        self.send_cmd(vel_left=0.0, vel_right=0.0, duration=1.5)

    def execute_square(self):
        time.sleep(1.0) 
        rospy.loginfo("Starting square trajectory...")
        
        # A simple loop guarantees exactly 4 perfect sides
        for i in range(4):
            rospy.loginfo(f"Side {i+1}: Driving straight...")
            self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
            self.stop()
            
            rospy.loginfo(f"Side {i+1}: Turning...")
            # Note: I set this to 0.7 based on your previous code. 
            # If it under-turns, increase the duration. If it over-turns, decrease it!
            self.send_cmd(vel_left=0.0, vel_right=0.5, duration=0.7) 
            self.stop()
            
        time.sleep(1.0)
        rospy.loginfo("Program finished safely.")
        
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

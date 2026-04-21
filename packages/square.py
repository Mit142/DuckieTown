#!/usr/bin/env python3
import os
import time  # <-- FIX 1: Imported Python's strict real-world time library
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot')
        topic_name = f"/{self.veh}/wheels_driver_node/wheels_cmd"
        
        # Queue size increased slightly to prevent dropped messages over Wi-Fi
        self.pub_cmd = rospy.Publisher(topic_name, WheelsCmdStamped, queue_size=10)

    def send_cmd(self, vel_left, vel_right, duration):
        """Publishes raw wheel commands for a set duration."""
        msg = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        
        # FIX 2: Use time.time() to grab the real-world start time
        start_time = time.time()
        
        while not rospy.is_shutdown() and (time.time() - start_time) < duration:
            # FIX 3: Update the timestamp every single loop so the robot knows it's a fresh command
            msg.header.stamp = rospy.Time.now()
            
            self.pub_cmd.publish(msg)
            
            # Strict real-world sleep for 0.1 seconds (10 Hz)
            time.sleep(0.1)
            
    def stop(self):
        """Halts the robot briefly to prevent drift before the next move."""
        self.send_cmd(vel_left=0.0, vel_right=0.0, duration=1.5)

    def execute_square(self):
        # Switched to time.sleep() for all pauses
        time.sleep(1.0) 
        
        rospy.loginfo("Starting square trajectory...")
        self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
        self.stop()
        self.send_cmd(vel_left=0, vel_right=0.5, duration=0.7) 
        self.stop()
        self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
        self.stop()
        self.send_cmd(vel_left=0, vel_right=0.5, duration=0.7) 
        self.stop()
        self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
        self.stop()
        self.send_cmd(vel_left=0, vel_right=0.5, duration=0.7) 
        self.stop()
        self.send_cmd(vel_left=0.7, vel_right=0.7, duration=1.5)
        self.stop()
        self.send_cmd(vel_left=0, vel_right=0.5, duration=0.7) 
        self.stop()
        
        # FIX 4: Added a final pause so the node doesn't close before the wheels actually stop
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

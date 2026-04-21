#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot')
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
            
    def stop(self, duration=0.5):
        """Halts the robot briefly to prevent drift."""
        self.send_cmd(vel_left=0.0, vel_right=0.0, duration=duration)

    def on_shutdown(self):
        self.stop(duration=0.1)

if __name__ == '__main__':
    node = SquareDriverNode(node_name="square_driver_node")
    
    try:
        rospy.sleep(1.0)
        rospy.loginfo("Starting square trajectory (Linear Mode)...")

        # --- SIDE 1 ---
        node.send_cmd(0.7, 0.7, 1.5) # Straight
        node.stop()
        node.send_cmd(-0.5, 0.5, 0.8) # Turn
        node.stop()

        # --- SIDE 2 ---
        node.send_cmd(0.7, 0.7, 1.5) # Straight
        node.stop()
        node.send_cmd(-0.5, 0.5, 0.8) # Turn
        node.stop()

        # --- SIDE 3 ---
        node.send_cmd(0.7, 0.7, 1.5) # Straight
        node.stop()
        node.send_cmd(-0.5, 0.5, 0.8) # Turn
        node.stop()

        # --- SIDE 4 ---
        node.send_cmd(0.7, 0.7, 1.5) # Straight
        node.stop()
        node.send_cmd(-0.5, 0.5, 0.8) # Turn
        
        # FINAL SAFETY STOP
        node.stop(duration=1.0)
        rospy.loginfo("Square finished.")

    except rospy.ROSInterruptException:
        pass


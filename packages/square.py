#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

class DriveSquareNode(DTROS):
    def __init__(self, node_name):
        super(DriveSquareNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        
        # Define our movement messages
        self.move_msg = WheelsCmdStamped(vel_left=0.4, vel_right=0.4)
        self.turn_msg = WheelsCmdStamped(vel_left=0.3, vel_right=-0.3) # Pivot turn
        self.stop_msg = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)

    def drive_duration(self, msg, duration):
        """Helper to publish a specific command for a set duration."""
        start_time = rospy.get_time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            self._publisher.publish(msg)
            rate.sleep()

    def run(self):
        rospy.loginfo("Starting the square pattern...")
        
        # A square has 4 sides and 4 turns
        for i in range(4):
            if rospy.is_shutdown():
                break
                
            # 1. Drive Straight
            rospy.loginfo(f"Side {i+1}: Driving straight")
            self.drive_duration(self.move_msg, 2.0) # Adjust for side length
            
            # 2. Stop briefly (helps with accuracy)
            self._publisher.publish(self.stop_msg)
            rospy.sleep(0.5)
            
            # 3. Turn 90 Degrees
            rospy.loginfo(f"Side {i+1}: Turning")
            self.drive_duration(self.turn_msg, 1.2) # Adjust this for a perfect 90°
            
            # 4. Stop briefly
            self._publisher.publish(self.stop_msg)
            rospy.sleep(0.5)
            
        rospy.loginfo("Square complete. Stopping!")
        self._publisher.publish(self.stop_msg)

    def on_shutdown(self):
        self._publisher.publish(self.stop_msg)

if __name__ == '__main__':
    node = DriveSquareNode(node_name='drive_square_node')
    node.run()



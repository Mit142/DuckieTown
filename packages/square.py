#!/usr/bin/env python3
import os
import time
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class SimpleCircleNode(DTROS):
    def __init__(self, node_name):
        super(SimpleCircleNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        self.veh = os.environ.get("VEHICLE_NAME", "duckiebot")
        
        # Publishing to the car_cmd_switch_node (which you confirmed works well!)
        topic_name = f"/{self.veh}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(topic_name, Twist2DStamped, queue_size=10)
        
        self.loginfo("Circle Node initialized and ready.")

    def drive_circle(self, duration_seconds):
        self.loginfo(f"Driving in a circle for {duration_seconds} seconds...")
        
        # The movement parameters (v = forward speed, omega = turn speed)
        speed = 0.2 
        turn_rate = 2.0 
        
        start_time = time.time()
        
        # Loop strictly based on real-world time
        while not rospy.is_shutdown() and (time.time() - start_time) < duration_seconds:
            msg = Twist2DStamped()
            msg.header.stamp = rospy.Time.now()  # Keep the safety switch happy
            msg.v = speed
            msg.omega = turn_rate
            
            self.pub_cmd.publish(msg)
            time.sleep(0.1)  # Publish at 10Hz
            
        self.loginfo("Circle maneuver complete!")

    def on_shutdown(self):
        """Called automatically when the node finishes or is killed."""
        self.loginfo("Shutting down... stopping motors.")
        
        # Send a final zero-velocity message to stop the robot
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = 0.0
        msg.omega = 0.0
        self.pub_cmd.publish(msg)
        
        # Brief pause to ensure the stop command reaches the robot over Wi-Fi
        time.sleep(1.0)
        super(SimpleCircleNode, self).on_shutdown()

if __name__ == '__main__':
    node = SimpleCircleNode(node_name='simple_circle_node')
    try:
        # Give the publisher a second to connect to the ROS network
        time.sleep(1.0) 
        
        # Tell the robot to drive in a circle for exactly 16 seconds
        node.drive_circle(duration_seconds=16.0)
        
    except rospy.ROSInterruptException:
        pass

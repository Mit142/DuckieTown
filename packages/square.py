#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        self.veh = rospy.get_namespace().strip("/")
        if not self.veh:
            self.veh = "duckiebot"

        # FIX: Publish to 'car_cmd_switch_node/wheels_cmd' or 'joy_mapper_node/car_cmd'
        # On 'ente' and newer versions, 'joy_mapper_node/car_cmd' is usually the most reliable
        # way to bypass the lane-following logic while still using the kinematics.
        topic_name = f"/{self.veh}/joy_mapper_node/car_cmd"
        
        self.pub_cmd = rospy.Publisher(topic_name, Twist2DStamped, queue_size=10)

    def send_cmd(self, v, omega, duration):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now() # Best practice: add a timestamp
        msg.v = v
        msg.omega = omega
        
        rate = rospy.Rate(10)
        end_time = rospy.Time.now() + rospy.Duration(duration)
        
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            # Update timestamp every loop
            msg.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(msg)
            rate.sleep()
            
    def stop(self):
        self.send_cmd(0.0, 0.0, 0.2)

    def execute_square(self):
        # Wait for the publisher to actually connect to the subscriber
        # If you publish immediately after creating the publisher, the message is lost.
        while self.pub_cmd.get_num_connections() < 1 and not rospy.is_shutdown():
            rospy.loginfo_once("Waiting for listener to connect...")
            rospy.sleep(0.1)

        rospy.loginfo("Starting square trajectory...")
        for i in range(4):
            rospy.loginfo(f"Side {i+1}: Driving straight...")
            self.send_cmd(v=0.4, omega=0.0, duration=2.0)
            self.stop()
            
            rospy.loginfo(f"Side {i+1}: Turning...")
            # Note: 1.57 is roughly 90 deg, but friction/battery will require tuning
            self.send_cmd(v=0.0, omega=2.5, duration=0.7) 
            self.stop()
            
        rospy.loginfo("Square completed.")

if __name__ == '__main__':
    node = SquareDriverNode(node_name="square_driver_node")
    try:
        node.execute_square()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class (Standard for 'ente')
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Get the vehicle name dynamically from the environment namespace
        self.veh = rospy.get_namespace().strip("/")
        if not self.veh:
            self.veh = "duckiebot" # Fallback if not launched with a standard namespace
            
        # Target the switch node to bypass the joystick and accept programmatic commands
        topic_name = f"/{self.veh}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(topic_name, Twist2DStamped, queue_size=10)

    def send_cmd(self, v, omega, duration):
        """Publishes a Twist2DStamped message for a set duration."""
        msg = Twist2DStamped()
        msg.v = v          # Linear velocity (m/s)
        msg.omega = omega  # Angular velocity (rad/s)
        
        rate = rospy.Rate(10) # 10 Hz publishing rate
        end_time = rospy.Time.now() + rospy.Duration(duration)
        
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.pub_cmd.publish(msg)
            rate.sleep()
            
    def stop(self):

          """Halts the robot briefly to prevent drift before the next move."""
        self.send_cmd(0.0, 0.0, 0.5)

    def execute_square(self):
        # Wait a moment for publishers to establish a connection
        rospy.sleep(1.0) 
        
        rospy.loginfo("Starting square trajectory...")
        for _ in range(4):
            # 1. Drive forward (v: speed, omega: 0)
            rospy.loginfo("Driving straight...")
            self.send_cmd(v=0.3, omega=0.0, duration=2.0)
            self.stop()
            
            # 2. Turn 90 degrees (v: 0, omega: turn speed)
            # 1.57 rad/s for 1 second is theoretically 90 degrees
            rospy.loginfo("Turning 90 degrees...")
            self.send_cmd(v=0.0, omega=1.57, duration=1.0) 
            self.stop()
            
        rospy.loginfo("Square completed.")

if __name__ == '__main__':
    node = SquareDriverNode(node_name="square_driver_node")
    try:
        node.execute_square()
    except rospy.ROSInterruptException:
        pass

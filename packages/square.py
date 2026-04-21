#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Get vehicle name from environment or default
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")
        
        # Publisher for car_cmd (v and omega)
        self.pub_car_cmd = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

    def publish_cmd(self, v, omega):
        """Helper to send velocity commands."""
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.pub_car_cmd.publish(msg)

    def drive_duration(self, v, omega, duration):
        """Drives at a specific velocity for a specific time."""
        t_end = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < t_end and not rospy.is_shutdown():
            self.publish_cmd(v, omega)
            rospy.sleep(0.1)  # 10Hz command rate
        
        # Brief stop between segments to reduce inertia/drift
        self.publish_cmd(0.0, 0.0)
        rospy.sleep(0.5)

    def execute_square(self):
        rospy.loginfo("Wait for 2 seconds before starting...")
        rospy.sleep(2.0)

        for i in range(4):
            rospy.loginfo(f"Executing Side {i+1}...")
            
            # 1. Drive Straight
            # v=0.4 (m/s approx), omega=0.0 (straight)
            self.drive_duration(0.4, 0.0, 1.5)
            
            # 2. Turn 90 Degrees Left
            # v=0.0 (turn in place), omega=2.5 (rad/s)
            # Adjust duration to get exactly 90 degrees!
            self.drive_duration(0.0, 2.5, 0.8)

        # Final Stop
        rospy.loginfo("Square complete. Stopping.")
        self.publish_cmd(0.0, 0.0)

    def on_shutdown(self):
        self.publish_cmd(0.0, 0.0)
        super(SquareDriverNode, self).on_shutdown()

if __name__ == '__main__':
    node = SquareDriverNode(node_name='square_driver_node')
    try:
        node.execute_square()
    except rospy.ROSInterruptException:
        pass



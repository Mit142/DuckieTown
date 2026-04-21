#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped


class SquareDriverNode(DTROS):
    def __init__(self, node_name):
        super(SquareDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
       
        # Use environment variable for flexibility, defaulting to your specific bot
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")
       
        # Taking inspiration from your circle code: using Twist2DStamped!
        topic_name = f"/{self.veh}/car_cmd_switch_node/cmd"
       
        self.pub_cmd = rospy.Publisher(topic_name, Twist2DStamped, queue_size=1)


    def publish_cmd(self, v, omega):
        """Helper to build and publish the Twist2DStamped message."""
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v          # Linear velocity (forward/backward)
        msg.omega = omega  # Angular velocity (turning)
        self.pub_cmd.publish(msg)


    def move_for_duration(self, v, omega, duration):
        """Publishes velocity commands for a set duration."""
        rate = rospy.Rate(10)
        start_time = rospy.get_time()
       
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            self.publish_cmd(v, omega)
            rate.sleep()
           
    def stop(self):
        """Halts the robot briefly to prevent drift before the next move."""
        self.publish_cmd(0.0, 0.0)
        rospy.sleep(0.5)


    def execute_square(self):
        rospy.sleep(1.0)
       
        self.log("Starting square trajectory...")
       
        # 4 Sides of the Square
        for i in range(4):
            # 1. Drive straight
            self.log(f"Side {i+1}: Driving straight...")
            # Note: You may need to tune 'v' and 'duration' for your specific robot's battery/motors
            self.move_for_duration(v=0.4, omega=0.0, duration=1.5)
            self.stop()
           
            # 2. Turn LEFT (90 degrees)
            self.log(f"Side {i+1}: Turning left...")
            # Note: You may need to tune 'omega' and 'duration' to get exactly 90 degrees
            self.move_for_duration(v=0.0, omega=1.5, duration=0.8)
            self.stop()
           
        self.log("Square completed!")
       
        # --- Safely shut down ---
        self.log("Halting motors...")
        self.stop()
       
        # This 1-second pause ensures the final 0.0 command is received before the node dies
        rospy.sleep(1.0)
        self.log("Program finished safely.")
       
    def on_shutdown(self):
        """Safety catch: Ensure wheels stop if the node is forcibly shut down (e.g., Ctrl+C)."""
        self.log("Shutting down... stopping motors.")
        self.publish_cmd(0.0, 0.0)
        super(SquareDriverNode, self).on_shutdown()


if __name__ == '__main__':
    node = SquareDriverNode(node_name="square_driver_node")
    try:
        node.execute_square()
    except rospy.ROSInterruptException:
        pass




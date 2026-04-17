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
            
        # FIX 1: We "spoof" the joystick topic. The robot's state machine 
        # is usually listening here by default and won't block the commands.
        topic_name = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_cmd = rospy.Publisher(topic_name, Twist2DStamped, queue_size=10)

    def send_cmd(self, v, omega, duration):
        """Publishes a Twist2DStamped message for a set duration."""
        msg = Twist2DStamped()
        msg.v = v          # Linear velocity (m/s)
        msg.omega = omega  # Angular velocity (rad/s)
        
        rate = rospy.Rate(10) # 10 Hz publishing rate
        end_time = rospy.Time.now() + rospy.Duration(duration)
        
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            # FIX 2: You MUST update the timestamp right before publishing!
            # If you don't do this, the robot ignores it as a "stale" safety hazard.
            msg.header.stamp = rospy.Time.now()
            
            self.pub_cmd.publish(msg)
            rate.sleep()
            
    def stop(self):
        """Halts the robot briefly to prevent drift before the next move."""
        self.send_cmd(0.0, 0.0, 0.5)

    def execute_square(self):
        # Wait a moment for publishers to establish a connection
        rospy.sleep(1.0) 
        
        rospy.loginfo("Starting square trajectory...")
        for i in range(4):
            # 1. Drive forward (v: speed, omega: 0)
            rospy.loginfo(f"Side {i+1}: Driving straight...")
            self.send_cmd(v=0.3, omega=0.0, duration=2.0)
            self.stop()
            
            # 2. Turn 90 degrees (v: 0, omega: turn speed)
            rospy.loginfo(f"Side {i+1}: Turning 90 degrees...")
            self.send_cmd(v=0.0, omega=1.57, duration=1.0) 
            self.stop()
            
        rospy.loginfo("Square completed.")

if __name__ == '__main__':
    node = SquareDriverNode(node_name="square_driver_node")
    try:
        node.execute_square()
    except rospy.ROSInterruptException:
        pass

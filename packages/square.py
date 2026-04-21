#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class CircleDriverNode(DTROS):
    def __init__(self, node_name):
        super(CircleDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        self.veh = rospy.get_namespace().strip("/")
        if not self.veh:
            self.veh = "duckiebot" 
            
        # Spoofing the joystick topic to bypass the FSM lock
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
            # Mandatory timestamp to pass the dead man's switch
            msg.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(msg)
            rate.sleep()
            
    def stop(self):
        """Halts the robot."""
        self.send_cmd(0.0, 0.0, 0.5)

    def execute_circle(self):
        # Wait a moment for publishers to establish a connection
        rospy.sleep(1.0) 
        
        rospy.loginfo("Starting circle trajectory...")
        
        # Drive forward AND turn at the same time for 10 seconds
        self.send_cmd(v=0.3, omega=0.6, duration=10.0) 
        
        self.stop()
        rospy.loginfo("Circle completed.")

if __name__ == '__main__':
    node = CircleDriverNode(node_name="circle_driver_node")
    try:
        node.execute_circle()
    except rospy.ROSInterruptException:
        pass

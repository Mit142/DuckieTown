#!/usr/bin/env python3
NODE_VERSION = "2.3.0"
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String

SQUARE_SIDES     = 4
FORWARD_SPEED    = 0.3    # m/s
FORWARD_DURATION = 2.0    # seconds per side — tune this
TURN_SPEED       = 1.5    # rad/s
TURN_DURATION    = 1.1    # seconds → 1.5 × 1.1 ≈ 1.65 rad ≈ 94°
STOP_DURATION    = 0.5    # pause between moves

class ShapeDriverNode(DTROS):
    def __init__(self, node_name):
        super(ShapeDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        self.veh = os.environ.get("VEHICLE_NAME", "duckiebot")
        self.pub_car_cmd = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        rospy.Subscriber(
            f"/{self.veh}/shape_driver_node/command",
            String,
            self.cb_command
        )
        
        # CHANGED: Set back to "square" so it runs automatically
        self.current_shape = "square" 
        self.log(f"Shape Driver v{NODE_VERSION} ready.")

    def cb_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ["square", "stop"]:
            self.current_shape = cmd
            self.log(f"Switching to: {cmd}")
        else:
            self.log(f"Unknown command: {cmd}")

    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = f"{self.veh}/base_link" 
        msg.v     = v
        msg.omega = omega
        self.pub_car_cmd.publish(msg)

    def drive_timed(self, v, omega, duration):
        """Publish at 20 Hz for the given duration, checking for interrupts."""
        t_end = rospy.Time


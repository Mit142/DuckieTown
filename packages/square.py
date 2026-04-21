#!/usr/bin/env python3

# Update NODE_VERSION on every change (see CLAUDE.md § Versioning)
NODE_VERSION = "1.3.0"

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String

CIRCLE_LAPS = 2
CIRCLE_LAP_DURATION = 8.0  # seconds per lap (tune to taste)


class ShapeDriverNode(DTROS):
    def __init__(self, node_name):
        super(ShapeDriverNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

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

        self.current_shape = "circle"
        self.log(f"Shape Driver v{NODE_VERSION} ready. Will do {CIRCLE_LAPS} circles then drive straight.")

    def cb_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ["circle", "straight", "stop"]:
            self.current_shape = cmd
            self.log(f"Switching to: {cmd}")
        else:
            self.log(f"Unknown command: {cmd}")

    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.pub_car_cmd.publish(msg)

    def drive_circles_then_straight(self):
        self.log("Starting circles...")
        for lap in range(CIRCLE_LAPS):
            if self.current_shape != "circle":
                return
            self.log(f"Circle lap {lap + 1}/{CIRCLE_LAPS}")
            t_end = rospy.Time.now() + rospy.Duration(CIRCLE_LAP_DURATION)
            while rospy.Time.now() < t_end:
                if self.current_shape != "circle":
                    return
                self.publish_cmd(0.2, 2.0)
                rospy.sleep(0.1)

        self.log("Circles done. Driving straight.")
        self.current_shape = "straight"

    def run(self):
        while not rospy.is_shutdown():
            if self.current_shape == "circle":
                self.drive_circles_then_straight()
            elif self.current_shape == "straight":
                self.publish_cmd(0.2, 0.0)
                rospy.sleep(0.1)
            else:
                self.publish_cmd(0.0, 0.0)
                rospy.sleep(0.1)

    def on_shutdown(self):
        self.publish_cmd(0.0, 0.0)
        super(ShapeDriverNode, self).on_shutdown()


if __name__ == '__main__':
    node = ShapeDriverNode(node_name='shape_driver_node')
    node.run()

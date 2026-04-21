#!/usr/bin/env python3

# Update NODE_VERSION on every change (see CLAUDE.md § Versioning)
NODE_VERSION = "1.4.0"

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String

# --- Square Configuration ---
SQUARE_LAPS = 2
STRAIGHT_DURATION = 2.0  # seconds to drive straight
STRAIGHT_SPEED = 0.3     # linear speed (m/s)
TURN_DURATION = 1.0      # seconds to turn (tune to hit exactly 90 degrees)
TURN_SPEED = 1.57        # angular speed (rad/s)
PAUSE_DURATION = 0.5     # brief stop between moves to prevent drift


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

        # Default to square
        self.current_shape = "square"
        self.log(f"Shape Driver v{NODE_VERSION} ready. Will do {SQUARE_LAPS} squares then drive straight.")

    def cb_command(self, msg):
        cmd = msg.data.strip().lower()
        # Updated to listen for 'square' instead of 'circle'
        if cmd in ["square", "straight", "stop"]:
            self.current_shape = cmd
            self.log(f"Switching to: {cmd}")
        else:
            self.log(f"Unknown command: {cmd}")

    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        # This timestamp is what keeps the Duckiebot from rejecting the command!
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.pub_car_cmd.publish(msg)

    def drive_squares_then_straight(self):
        self.log("Starting squares...")
        for lap in range(SQUARE_LAPS):
            if self.current_shape != "square":
                return
            self.log(f"Square lap {lap + 1}/{SQUARE_LAPS}")

            # 4 sides to a square
            for side in range(4):
                if self.current_shape != "square":
                    return

                # 1. Drive Straight
                self.log(f"  Side {side + 1}: Driving straight")
                t_end = rospy.Time.now() + rospy.Duration(STRAIGHT_DURATION)
                while rospy.Time.now() < t_end:
                    if self.current_shape != "square": 
                        return
                    self.publish_cmd(STRAIGHT_SPEED, 0.0)
                    rospy.sleep(0.1)

                # 2. Pause
                self.publish_cmd(0.0, 0.0)
                rospy.sleep(PAUSE_DURATION)

                # 3. Turn 90 Degrees
                self.log(f"  Side {side + 1}: Turning")
                t_end = rospy.Time.now() + rospy.Duration(TURN_DURATION)
                while rospy.Time.now() < t_end:
                    if self.current_shape != "square": 
                        return
                    self.publish_cmd(0.0, TURN_SPEED)
                    rospy.sleep(0.1)

                # 4. Pause
                self.publish_cmd(0.0, 0.0)
                rospy.sleep(PAUSE_DURATION)

        self.log("Squares done. Driving straight.")
        self.current_shape = "straight"

    def run(self):
        while not rospy.is_shutdown():
            if self.current_shape == "square":
                self.drive_squares_then_straight()
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

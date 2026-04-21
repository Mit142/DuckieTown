#!/usr/bin/env python3
# Update NODE_VERSION on every change (see CLAUDE.md § Versioning)
NODE_VERSION = "2.0.0"
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String

SQUARE_SIDES        = 4
FORWARD_SPEED       = 0.2    # m/s — same as your circle code
FORWARD_DURATION    = 2.0    # seconds per side (tune to taste)
TURN_SPEED          = 2.0    # rad/s — same omega as your circle code
TURN_DURATION       = 0.785  # seconds → 2.0 × 0.785 ≈ 1.57 rad ≈ 90°
STOP_DURATION       = 0.3    # brief pause between moves

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
        self.current_shape = "square"
        self.log(f"Shape Driver v{NODE_VERSION} ready. Will drive in a square.")

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
        msg.v     = v
        msg.omega = omega
        self.pub_car_cmd.publish(msg)

    def drive_timed(self, v, omega, duration):
        """Publish a command for a fixed duration, checking for shutdown."""
        t_end = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < t_end:
            if rospy.is_shutdown() or self.current_shape != "square":
                return
            self.publish_cmd(v, omega)
            rospy.sleep(0.1)

    def drive_square(self):
        self.log("Starting square...")
        for side in range(SQUARE_SIDES):
            if rospy.is_shutdown() or self.current_shape != "square":
                return

            # 1. Drive straight
            self.log(f"Side {side + 1}/{SQUARE_SIDES} — driving forward")
            self.drive_timed(FORWARD_SPEED, 0.0, FORWARD_DURATION)

            # 2. Stop briefly
            self.drive_timed(0.0, 0.0, STOP_DURATION)

            # 3. Turn 90°
            self.log(f"Side {side + 1}/{SQUARE_SIDES} — turning 90°")
            self.drive_timed(0.0, TURN_SPEED, TURN_DURATION)

            # 4. Stop briefly
            self.drive_timed(0.0, 0.0, STOP_DURATION)

        self.log("Square complete! Stopping.")
        self.current_shape = "stop"

    def run(self):
        while not rospy.is_shutdown():
            if self.current_shape == "square":
                self.drive_square()
            elif self.current_shape == "stop":
                self.publish_cmd(0.0, 0.0)
                rospy.sleep(0.1)

    def on_shutdown(self):
        self.publish_cmd(0.0, 0.0)
        super(ShapeDriverNode, self).on_shutdown()

if __name__ == '__main__':
    node = ShapeDriverNode(node_name='shape_driver_node')
    node.run()



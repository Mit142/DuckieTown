#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import time

# --- Configuration ---
VEHICLE_NAME = "entebot208"

# Forward speed (m/s) and duration (s)
LINEAR_SPEED  = 0.3   # adjust to your surface/wheel calibration
FORWARD_TIME  = 2.0   # seconds to drive one side of the square

# Turn speed (rad/s) and duration (s)
# For a 90° turn: angular_speed × turn_time ≈ π/2 ≈ 1.5708 rad
ANGULAR_SPEED = 0.785  # rad/s  (π/4)
TURN_TIME     = 2.0    # seconds  → 0.785 × 2.0 ≈ 1.57 rad ≈ 90°

PAUSE_TIME    = 0.5    # brief stop between moves


def drive_square():
    rospy.init_node("drive_square_node", anonymous=True)

    # Topic: /<vehicle>/car_node/cmd_vel  (adjust if your setup differs)
    topic = f"/{VEHICLE_NAME}/car_node/cmd_vel"
    pub   = rospy.Publisher(topic, Twist, queue_size=10)

    rospy.loginfo(f"Publishing to: {topic}")
    rospy.sleep(1.0)   # wait for publisher to register

    rate = rospy.Rate(10)  # 10 Hz

    for side in range(4):
        if rospy.is_shutdown():
            break

        # ── 1. Drive forward ──────────────────────────────
        rospy.loginfo(f"Side {side + 1}/4 — driving forward")
        msg_forward        = Twist()
        msg_forward.linear.x  = LINEAR_SPEED
        msg_forward.angular.z = 0.0

        t_start = time.time()
        while time.time() - t_start < FORWARD_TIME and not rospy.is_shutdown():
            pub.publish(msg_forward)
            rate.sleep()

        # ── 2. Stop ───────────────────────────────────────
        pub.publish(Twist())   # all-zero Twist = stop
        rospy.sleep(PAUSE_TIME)

        # ── 3. Turn 90° left ─────────────────────────────
        rospy.loginfo(f"Side {side + 1}/4 — turning 90°")
        msg_turn           = Twist()
        msg_turn.linear.x  = 0.0
        msg_turn.angular.z = ANGULAR_SPEED   # positive = left / CCW

        t_start = time.time()
        while time.time() - t_start < TURN_TIME and not rospy.is_shutdown():
            pub.publish(msg_turn)
            rate.sleep()

        # ── 4. Stop ───────────────────────────────────────
        pub.publish(Twist())
        rospy.sleep(PAUSE_TIME)

    rospy.loginfo("Square complete!")


if __name__ == "__main__":
    try:
        drive_square()
    except rospy.ROSInterruptException:
        pass



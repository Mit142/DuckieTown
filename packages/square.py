import rospy
from duckietown_msgs.msg import Twist2DStamped
import time

VEHICLE_NAME  = "entebot208"

LINEAR_SPEED  = 0.3    # m/s
FORWARD_TIME  = 2.0    # seconds per side

ANGULAR_SPEED = 0.785  # rad/s (π/4)
TURN_TIME     = 2.0    # seconds → ~90°

PAUSE_TIME    = 0.5


def stop(pub):
    msg = Twist2DStamped()
    msg.v     = 0.0
    msg.omega = 0.0
    pub.publish(msg)
    rospy.sleep(PAUSE_TIME)


def drive_square():
    rospy.init_node("drive_square_node", anonymous=True)

    # Correct Duckietown daffy topic
    topic = f"/{VEHICLE_NAME}/kinematics_node/velocity"
    pub   = rospy.Publisher(topic, Twist2DStamped, queue_size=10)

    rospy.loginfo(f"Publishing to: {topic}")
    rospy.sleep(1.0)

    rate = rospy.Rate(10)

    for side in range(4):
        if rospy.is_shutdown():
            break

        # ── 1. Drive forward ──────────────────────────────
        rospy.loginfo(f"Side {side + 1}/4 — driving forward")
        msg            = Twist2DStamped()
        msg.v          = LINEAR_SPEED
        msg.omega      = 0.0

        t_start = time.time()
        while time.time() - t_start < FORWARD_TIME and not rospy.is_shutdown():
            pub.publish(msg)
            rate.sleep()

        stop(pub)

        # ── 2. Turn 90° ───────────────────────────────────
        rospy.loginfo(f"Side {side + 1}/4 — turning 90°")
        msg            = Twist2DStamped()
        msg.v          = 0.0
        msg.omega      = ANGULAR_SPEED

        t_start = time.time()
        while time.time() - t_start < TURN_TIME and not rospy.is_shutdown():
            pub.publish(msg)
            rate.sleep()

        stop(pub)

    rospy.loginfo("Square complete!")


if __name__ == "__main__":
    try:
        drive_square()
    except rospy.ROSInterruptException:
        pass


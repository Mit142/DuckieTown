#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time

def move_square():
    rospy.init_node('square_driver')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    move_cmd = Twist()

    # Parameters (tune these for your robot)
    forward_speed = 0.2   # meters per second
    turn_speed = 0.5      # radians per second

    forward_time = 2.0    # time to move one side
    turn_time = 1.57 / turn_speed  # time to turn ~90 degrees

    rospy.sleep(1)  # wait for publisher to connect

    for i in range(4):
        # Move forward
        move_cmd.linear.x = forward_speed
        move_cmd.angular.z = 0.0
        pub.publish(move_cmd)
        time.sleep(forward_time)

        # Stop briefly
        move_cmd.linear.x = 0.0
        pub.publish(move_cmd)
        time.sleep(0.5)

        # Turn 90 degrees
        move_cmd.angular.z = turn_speed
        pub.publish(move_cmd)
        time.sleep(turn_time)

        # Stop briefly again
        move_cmd.angular.z = 0.0
        pub.publish(move_cmd)
        time.sleep(0.5)

    # Final stop
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 0.0
    pub.publish(move_cmd)

if __name__ == '__main__':
    try:
        move_square()
    except rospy.ROSInterruptException:
        pass


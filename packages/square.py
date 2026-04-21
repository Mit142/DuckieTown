import rospy
from geometry_msgs.msg import Twist
import math

def move_square():
    # 1. Initialize the ROS node
    rospy.init_node('duckiebot_square_node', anonymous=True)

    # 2. Set up the Publisher
    # IMPORTANT: Replace 'duckiebot_name' with your actual Duckiebot's hostname
    vel_pub = rospy.Publisher('/duckiebot_name/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10) # Publish at 10 Hz

    # 3. Define movement parameters
    linear_speed = 0.2  # meters per second
    angular_speed = 0.5 # radians per second
    
    # Time = Distance / Speed
    side_duration = 3.0 # Drive straight for 3 seconds
    
    # 90 degrees in radians is pi/2. Time = (pi/2) / angular_speed
    turn_duration = (math.pi / 2) / angular_speed

    # Create empty Twist messages for movement and stopping
    move_cmd = Twist()
    move_cmd.linear.x = linear_speed
    move_cmd.angular.z = 0.0

    turn_cmd = Twist()
    turn_cmd.linear.x = 0.0
    turn_cmd.angular.z = angular_speed

    stop_cmd = Twist() # Default is all zeros

    rospy.sleep(1) # Give the node a second to connect to the publisher

    # 4. Execute the square loop 4 times
    for i in range(4):
        rospy.loginfo(f"Moving straight: Side {i+1}")
        start_time = rospy.Time.now().to_sec()
        # Drive forward
        while rospy.Time.now().to_sec() - start_time < side_duration:
            vel_pub.publish(move_cmd)
            rate.sleep()

        # Brief stop to prevent drifting
        vel_pub.publish(stop_cmd)
        rospy.sleep(0.5)

        rospy.loginfo("Turning 90 degrees")
        start_time = rospy.Time.now().to_sec()
        # Turn
        while rospy.Time.now().to_sec() - start_time < turn_duration:
            vel_pub.publish(turn_cmd)
            rate.sleep()

        # Brief stop before the next side
        vel_pub.publish(stop_cmd)
        rospy.sleep(0.5)

    # Final stop after the square is complete
    vel_pub.publish(stop_cmd)
    rospy.loginfo("Square completed!")

if __name__ == '__main__':
    try:
        move_square()
    except rospy.ROSInterruptException:
        pass















import time
from duckietown_msgs.msg import Twist2DStamped

# Constants for movement
LINEAR_SPEED = 0.5  # m/s
ANGULAR_SPEED = 1.5 # rad/s
MOVE_DURATION = 2.0 # seconds to move forward
TURN_DURATION = 1.0 # adjust this until the turn is exactly 90 degrees

def drive_square(publisher):
    # Prepare the message
    msg = Twist2DStamped()

    for i in range(4):
        # 1. Move Forward
        print(f"Side {i+1}: Moving forward...")
        msg.v = LINEER_SPEED
        msg.omega = 0.0
        publisher.publish(msg)
        time.sleep(MOVE_DURATION)

        # 2. Stop briefly to stabilize
        msg.v = 0.0
        msg.omega = 0.0
        publisher.publish(msg)
        time.sleep(0.5)

        # 3. Turn Left 90 Degrees
        print(f"Side {i+1}: Turning left...")
        msg.v = 0.0
        msg.omega = ANGULAR_SPEED
        publisher.publish(msg)
        time.sleep(TURN_DURATION)

        # 4. Stop before next side
        msg.v = 0.0
        msg.omega = 0.0
        publisher.publish(msg)
        time.sleep(0.5)

    print("Square completed!")


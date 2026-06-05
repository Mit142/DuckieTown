#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------
source /opt/ros/noetic/setup.bash

# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)
export ROS_MASTER_URI=http://entebot208.local:11311
export ROS_IP=$(hostname -I | awk '{print $1}')
# launching app
python3 $DT_REPO_PATH/packages/circle.py

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join

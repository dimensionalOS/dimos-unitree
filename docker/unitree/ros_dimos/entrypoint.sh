#!/bin/bash
set -e

# Create supervisor log directory
mkdir -p /var/log/supervisor

# Source ROS2 environment
source /opt/ros/${ROS_DISTRO}/setup.bash
source /ros2_ws/install/setup.bash
# Execute the command passed to docker run
exec "$@"

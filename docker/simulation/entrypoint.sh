#!/bin/bash
cd /isaac-sim && ./setup_python_env.sh
export PYTHONPATH="${PYTHONPATH}:/app"
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
exec "$@" 
#!/bin/bash

# Check if an argument was provided
if [ $# -gt 0 ]; then
  option=$1
else
  echo "Select an option:"
  echo "1) Docker compose sequence: Takes down containers, builds, then brings them up."
  echo "2) Attach to tmux session: Exec into the container and attach to the 'python_session'."
  read -p "Enter option (1 or 2): " option
fi

case $option in
  1)
    docker compose -f ./docker/unitree/ros_agents/docker-compose.yml down && \
    docker compose -f ./docker/unitree/ros_agents/docker-compose.yml build && \
    docker compose -f ./docker/unitree/ros_agents/docker-compose.yml up
    ;;
  2)
    docker exec -it ros_agents-dimos-unitree-ros-agents-1 tmux attach-session -t python_session
    ;;
  *)
    echo "Invalid option. Please run the script again and select 1 or 2."
    ;;
esac

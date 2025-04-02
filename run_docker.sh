#!/bin/bash

# Function to display menu options
show_menu() {
  echo "🐳 DIMOS Docker Runner 🐳"
  echo "=================================="
  echo "Available commands:"
  echo "  1 | run        : Build and run ros_agents containers"
  echo "  2 | attach     : Attach to tmux session in the running container"
  echo "  3 | rebuild    : Full rebuild of containers (--no-cache)"
  echo "  4 | web        : Build and run web-os container"
  echo "  5 | agent      : Build and run agent container"
  echo "  6 | seg        : Build and run semantic-seg model container"
  echo "  7 | seg-robot  : Build and run semantic-seg robot container"
  echo "  8 | huggingface: Build and run huggingface local model"
  echo "  9 | huggingface-remote: Build and run huggingface remote model"
  echo "=================================="
}

# Function to run docker compose commands
run_docker_compose() {
  local file=$1
  local rebuild=$2

  if [ "$rebuild" = "full" ]; then
    echo "📦 Full rebuild with --no-cache..."
    docker compose -f $file down --rmi all -v && \
    docker compose -f $file build --no-cache && \
    docker compose -f $file up
  else
    echo "🚀 Building and running containers..."
    docker compose -f $file down && \
    docker compose -f $file build && \
    docker compose -f $file up
  fi
}

# Check if an argument was provided
if [ $# -gt 0 ]; then
  option=$1
else
  show_menu
  read -p "Enter option (number or command): " option
fi

# Process the option - support both numbers and text commands
case $option in
  1|run)
    run_docker_compose "./docker/unitree/ros_agents/docker-compose.yml"
    ;;
  2|attach)
    echo "🔌 Attaching to tmux session..."
    docker exec -it ros_agents-dimos-unitree-ros-agents-1 tmux attach-session -t python_session
    ;;
  3|rebuild)
    run_docker_compose "./docker/unitree/ros_agents/docker-compose.yml" "full"
    ;;
  4|web)
    run_docker_compose "./docker/interface/docker-compose.yml"
    ;;
  5|agent)
    run_docker_compose "./docker/agent/docker-compose.yml"
    ;;
  6|seg)
    run_docker_compose "./docker/models/semantic_seg/docker-compose.yml"
    ;;
  7|seg-robot)
    run_docker_compose "./docker/unitree/ros_dimos_seg/docker-compose.yml"
    ;;
  8|huggingface)
    run_docker_compose "./docker/models/huggingface_local/docker-compose.yml"
    ;;
  9|huggingface-remote)
    run_docker_compose "./docker/models/huggingface_remote/docker-compose.yml"
    ;;
  help|--help|-h)
    show_menu
    ;;
  *)
    echo "❌ Invalid option: $option"
    show_menu
    exit 1
    ;;
esac
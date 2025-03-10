![Screenshot 2025-02-18 at 16-31-22 DimOS Terminal](/assets/dimos_terminal.png)

# The Dimensional Framework
*The universal framework for AI-native generalist robotics.*

## What is Dimensional?

Dimensional is an open-source framework for building agentive generalist robots. It is designed to be a modular, extensible, and scalable framework for building robots that can learn and improve over time.

## DIMOS x Unitree Go2

We are shipping a first look at the DIMOS x Unitree Go2 integration, allowing for seamless incorporation of off-the-shelf Agents() with Unitree ROS2 Nodes and WebRTC action primitives, including:

- Navigation control primitives (move, reverse, spinLeft, spinRight, etc.)
- WebRTC control primitives (FrontPounce, FrontFlip, FrontJump, etc.)
- Camera feeds (image_raw, compressed_image, etc.)
- IMU data
- State information
- Lidar / PointCloud data 🚧
- Any other generic Unitree ROS2 topics

### Features 

- **Agentive AI**
  - Agent() classes with planning, spatial reasoning, and Robot.Skill() tool calling abilities.
  - Integrate with any off-the-shelf model: OpenAIAgent, GeminiAgent 🚧, DeepSeekAgent 🚧, HuggingfaceAgent 🚧, etc.
  - Modular agent architecture for easy extensibility and chaining of Agent output --> Subagents input. 
  - Agent spatial / language memory for location grounded reasoning and recall. 🚧

- **DimOS Infrastructure**
  - Pub/Sub architecture powered by ReactiveX for real-time data streaming between ROS2 and Agents.
  - A reactive data streaming architecture using RxPY to manage real-time video (or other sensor input), outbound commands, and inbound robot state between the DimOS Development interface, Agents, and ROS2.

- **DimOS Development Tools**
  - FastAPI-based web interface
  - Comprehensive testing framework
  - Docker containerization support
  - Terraform infrastructure as code

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dimensional.git
cd dimensional

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp default.env .env
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- A Unitree Go2 robot accessible on your network
- The robot's IP address
- OpenAI API Key

### Configuration:

Configure your environment variables in `.env`
```bash
OPENAI_API_KEY=<OPENAI_API_KEY>
ROBOT_IP=<ROBOT_IP>
CONN_TYPE=webrtc
WEBRTC_SERVER_HOST=0.0.0.0
WEBRTC_SERVER_PORT=9991
DISPLAY=:0

# Optional
DIMOS_MAX_WORKERS=
ROS_OUTPUT_DIR=/app/assets/output/ros
```

### Run via Docker 
```bash
xhost +local:root # If running locally and desire RVIZ GUI
docker compose -f docker/unitree/ros_agents/docker-compose.yml up --build
```

```
dimos/
├── agents/      # Agent implementation and behaviors
├── robot/       # Robot control and hardware interface
├── stream/      # WebRTC and data streaming
├── web/         # Web interface and API
├── simulation/  # Robot simulation environment
├── utils/       # Utility functions and helpers
└── types/       # Type definitions and interfaces
```

## Documentation

For detailed documentation, please visit our [documentation site](#) (Coming Soon).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Unitree Robotics for their Go2 platform
- The open-source robotics community
- All contributors and supporters of the Dimensional Framework

## Contact

- GitHub Issues: For bug reports and feature requests
- Email: [your-email@example.com]
- Twitter: [@DimensionalAI]


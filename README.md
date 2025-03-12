![Screenshot 2025-02-18 at 16-31-22 DimOS Terminal](/assets/dimos_terminal.png)

# The Dimensional Framework
*The universal framework for AI-native generalist robotics*

## What is Dimensional?

Dimensional is an open-source framework for building agentive generalist robots. DimOS allows off-the-shelf Agents to call tools/functions and read sensor/state data directly from ROS. 

The framework enables orchestration of Agents() as generalized spatial planners and robot skills/action primitives as functions. 

The result: cross-embodied *"Dimensional Applications"* exceptional at generalization and robust at symbolic action execution. 

## DIMOS x Unitree Go2

We are shipping a first look at the DIMOS x Unitree Go2 integration, allowing for off-the-shelf Agents() to "call" Unitree ROS2 Nodes and WebRTC action primitives, including:

- Navigation control primitives (move, reverse, spinLeft, spinRight, etc.)
- WebRTC control primitives (FrontPounce, FrontFlip, FrontJump, etc.)
- Camera feeds (image_raw, compressed_image, etc.)
- IMU data
- State information
- Lidar / PointCloud primitives 🚧
- Any other generic Unitree ROS2 topics

### Features 

- **DimOS Agents**
  - Agent() classes with planning, spatial reasoning, and Robot.Skill() function calling abilities.
  - Integrate with any off-the-shelf model: OpenAIAgent, GeminiAgent 🚧, DeepSeekAgent 🚧, HuggingfaceAgent 🚧, etc.
  - Modular agent architecture for easy extensibility and chaining of Agent output --> Subagents input. 
  - Agent spatial / language memory for location grounded reasoning and recall. 🚧

- **DimOS Infrastructure**
  - A reactive data streaming architecture using RxPY to manage real-time video (or other sensor input), outbound commands, and inbound robot state between the DimOS interface, Agents, and ROS2.
  - Robot Command Queue to handle complex multi-step actions to Robot.  
  - Simulation bindings (Genesis, Isaacsim, etc.) to test your agentive application before deploying to a physical robot. 

- **DimOS Interface / Development Tools**
  - Local development interface to control your robot, orchestrate agents, visualize camera/lidar streams, and debug your dimensional agentive application.  


## Installation

```bash
# Clone the repository
git clone https://github.com/dimensionalOS/dimos-unitree.git
cd dimos-unitree

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp default.env .env
```

## Quick Start 🚀

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
```

### Run via Docker 
```bash
xhost +local:root # If running locally and desire RVIZ GUI
docker compose -f docker/unitree/ros_agents/docker-compose.yml up --build # TODO: change docker path
```

### Project Structure 

Non-production directories excluded 
```
.
├── dimos/
│   ├── agents/      # Agent implementation and behaviors
│   ├── robot/       # Robot control and hardware interface
│   │   └── unitree/ # Unitree Go2 specific control and skills implementations
│   ├── stream/      # WebRTC and data streaming
│   ├── web/         # DimOS development interface and API
│   │   └── dimos_interface/  # DimOS web interface 
│   ├── simulation/  # Robot simulation environments
│   ├── utils/       # Utility functions and helpers
│   └── types/       # Type definitions and interfaces
├── tests/           # Test files
└── docker/          # Docker configuration files and compose definitions
```

## Building


### Simple DimOS Application

```python
from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from dimos.agents.agent import OpenAIAgent

# Initialize robot
robot = UnitreeGo2(ip=robot_ip,
                  ros_control=UnitreeROSControl(),
                  skills=MyUnitreeSkills())

# Initialize agent
agent = OpenAIAgent(
            dev_name="UnitreeExecutionAgent",
            input_video_stream=robot.get_ros_video_stream(),
            skills=robot.get_skills(),
            system_query="Jump when you see a human! Front flip when you see a dog!",
            model_name="gpt-4o"
        )

while True: # keep process running
  time.sleep(1)
```


### DimOS Application with Agent chaining

Let's build a simple DimOS application with Agent chaining. We define a ```planner``` as a ```PlanningAgent``` that takes in user input to devise a complex multi-step plan. This plan is passed step-by-step to an ```executor``` agent that can queue ```AbstractSkill``` commands to the ```ROSCommandQueue```. 

Our reactive Pub/Sub data streaming architecture allows for chaining of ```Agent_0 --> Agent_1 --> ... --> Agent_n``` via the ```input_query_stream``` parameter in each which takes an ```Observable``` input from the previous Agent in the chain. 

**Via this method you can chain together any number of Agents() to create complex dimensional applications.** 

```python

web_interface = RobotWebInterface(port=5555)

robot = UnitreeGo2(ip=robot_ip,
                  ros_control=UnitreeROSControl(),
                  skills=MyUnitreeSkills())

# Initialize master planning agent
planner = PlanningAgent(
            dev_name="UnitreePlanningAgent",
            input_query_stream=web_interface.query_stream, # Takes user input from dimOS interface
            skills=robot.get_skills(),
            model_name="gpt-4o",
        )

# Initialize execution agent
executor = OpenAIAgent(
            dev_name="UnitreeExecutionAgent",
            input_query_stream=planner.get_response_observable(), # Takes planner output as input
            skills=robot.get_skills(),
            model_name="gpt-4o",
            system_query="""
            You are a robot execution agent that can execute tasks on a virtual
            robot. ONLY OUTPUT THE SKILLS TO EXECUTE.
            """
        )

while True: # keep process running
  time.sleep(1)
```

### Calling Action Primitives

Call action primitives directly from ```Robot()``` for prototyping and testing.

```python
robot = UnitreeGo2(ip=robot_ip,)

# Call a Unitree WebRTC action primitive
robot.webrtc_req(api_id=1016) # "Hello" command

# Call a ROS2 action primitive
robot.move(distance=1.0, speed=0.5)
```

### Creating Custom Skills (non-unitree specific)

#### Create basic custom skills by inheriting from ```AbstractSkill``` and implementing the ```__call__``` method.

```python
class Flip(AbstractSkill):
    def __call__(self, robot):
      return self.robot.flip(robot)
```

#### Chain together skills to create recursive skill trees

```python
class Jump(AbstractSkill):
    def __call__(self):
      return self.robot.jump()

class Flip(AbstractSkill):
    def __call__(self):
      return self.robot.flip()

class JumpAndFlip(AbstractSkill):
    def __init__(self, robot):
        super().__init__(robot)
        self.jump = Jump(robot)
        self.flip = Flip(robot)

    def __call__(self):
        return [self.jump(), self.flip()]
```

## Documentation

For detailed documentation, please visit our [documentation site](#) (Coming Soon).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) (Coming soon) for details on how to get started.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Huge thanks to! 
- The Roboverse Community and their unitree-specific help. Check out their [Discord](https://discord.gg/HEXNMCNhEh). 
- @abizovnuralem for his work on the [Unitree Go2 ROS2 SDK](https://github.com/abizovnuralem/go2_ros2_sdk) we integrate with for DimOS. 
- @legion1581 for his work on the [Unitree Go2 WebRTC Connect](https://github.com/legion1581/go2_webrtc_connect) from which we've pulled the ```Go2WebRTCConnection``` class and other types for seamless WebRTC-only integration with DimOS. 
- @tfoldi for the webrtc_req integration via Unitree Go2 ROS2 SDK, which allows for seamless usage of Unitree WebRTC control primitives with DimOS. 

## Contact

- GitHub Issues: For bug reports and feature requests
- Email: [build@dimensionalOS.com](mailto:build@dimensionalOS.com)


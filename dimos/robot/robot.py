from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field
from dimos.hardware.interface import HardwareInterface
from dimos.agents.agent_config import AgentConfig
from dimos.robot.ros_control import ROSControl
from dimos.robot.skills import AbstractSkill

'''
Base class for all dimos robots, both physical and simulated.
'''
class Robot(ABC):
    def __init__(self,
                 agent_config: AgentConfig = None,
                 hardware_interface: HardwareInterface = None,
                 ros_control: ROSControl = None):
        
        self.agent_config = agent_config
        self.hardware_interface = hardware_interface
        self.ros_control = ros_control

    def move(self, x: float, y: float, yaw: float, duration: float = 0.0) -> bool:
        """Move the robot using velocity commands.
        
        Args:
            x: Forward/backward velocity (m/s)
            y: Left/right velocity (m/s)
            yaw: Rotational velocity (rad/s)
            duration: How long to move (seconds). If 0, command is continuous
            
        Returns:
            bool: True if command was sent successfully
        """
        if self.ros_control is None:
            raise RuntimeError("No ROS control interface available for movement")
        return self.ros_control.move(x, y, yaw, duration)

    @abstractmethod
    def do(self, *args, **kwargs):
     """Executes motion."""
    pass
    def update_hardware_interface(self, new_hardware_interface: HardwareInterface):
        """Update the hardware interface with a new configuration."""
        self.hardware_interface = new_hardware_interface

    def get_hardware_configuration(self):
        """Retrieve the current hardware configuration."""
        return self.hardware_interface.get_configuration()

    def set_hardware_configuration(self, configuration):
        """Set a new hardware configuration."""
        self.hardware_interface.set_configuration(configuration)


class MyUnitreeSkills(AbstractSkill):
    """My Unitree Skills."""

    _robot: Optional[Robot] = None

    def __init__(self, robot: Optional[Robot] = None, **data):
        super().__init__(**data)
        self._robot: Robot = robot

    class Move(AbstractSkill):
        """Move the robot using velocity commands."""

        _robot: Robot = None
        _MOVE_PRINT_COLOR: str = "\033[32m"
        _MOVE_RESET_COLOR: str = "\033[0m"

        x: float = Field(..., description="Forward/backward velocity (m/s)")
        y: float = Field(..., description="Left/right velocity (m/s)")
        yaw: float = Field(..., description="Rotational velocity (rad/s)")
        duration: float = Field(..., description="How long to move (seconds). If 0, command is continuous")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(f"{self._MOVE_PRINT_COLOR}Initializing Move Skill{self._MOVE_RESET_COLOR}")
            self._robot = robot
            print(f"{self._MOVE_PRINT_COLOR}Move Skill Initialized with Robot: {self._robot}{self._MOVE_RESET_COLOR}")

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Move Skill")
            elif self._robot.ros_control is None:
                raise RuntimeError("No ROS control interface available for movement")
            else:
                return self._robot.ros_control.move(self.x, self.y, self.yaw, self.duration)
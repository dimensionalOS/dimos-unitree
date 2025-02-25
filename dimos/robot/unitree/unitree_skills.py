
from typing import Optional

from pydantic import Field
from dimos.robot.robot import Robot
from dimos.robot.skills import AbstractSkill

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

    class Wave(AbstractSkill):
        """Wave the hand of therobot."""

        _robot: Robot = None
        _WAVE_PRINT_COLOR: str = "\033[32m"
        _WAVE_RESET_COLOR: str = "\033[0m"

        duration: float = Field(..., description="How long to wave (seconds). If 0, command is continuous")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(f"{self._WAVE_PRINT_COLOR}Initializing Wave Skill{self._WAVE_RESET_COLOR}")
            self._robot = robot
            print(f"{self._WAVE_PRINT_COLOR}Wave Skill Initialized with Robot: {self._robot}{self._WAVE_RESET_COLOR}")

        def __call__(self):
            return "Wave was successful."

from typing import List, Optional, Tuple, Type
import time
from pydantic import Field
from dimos.robot.robot import Robot
from dimos.robot.skills import AbstractSkill

# Module-level constant for Unitree ROS control definitions
UNITREE_ROS_CONTROLS: List[Tuple[str, int, str]] = [
    ("Damp", 1001, ""),
    ("BalanceStand", 1002, ""),
    ("StopMove", 1003, ""),
    ("StandUp", 1004, ""),
    ("StandDown", 1005, ""),
    ("RecoveryStand", 1006, ""),
    ("Euler", 1007, ""),
    # ("Move", 1008, "Move the robot using velocity commands."),  # Intentionally omitted
    ("Sit", 1009, ""),
    ("RiseSit", 1010, ""),
    ("SwitchGait", 1011, ""),
    ("Trigger", 1012, ""),
    ("BodyHeight", 1013, ""),
    ("FootRaiseHeight", 1014, ""),
    ("SpeedLevel", 1015, ""),
    ("Hello", 1016, ""),
    ("Stretch", 1017, ""),
    ("TrajectoryFollow", 1018, ""),
    ("ContinuousGait", 1019, ""),
    ("Content", 1020, ""),
    ("Wallow", 1021, ""),
    ("Dance1", 1022, ""),
    ("Dance2", 1023, ""),
    ("GetBodyHeight", 1024, ""),
    ("GetFootRaiseHeight", 1025, ""),
    ("GetSpeedLevel", 1026, ""),
    ("SwitchJoystick", 1027, ""),
    ("Pose", 1028, ""),
    ("Scrape", 1029, ""),
    ("FrontFlip", 1030, ""),
    ("FrontJump", 1031, ""),
    ("FrontPounce", 1032, ""),
    ("WiggleHips", 1033, ""),
    ("GetState", 1034, ""),
    ("EconomicGait", 1035, ""),
    ("FingerHeart", 1036, ""),
    ("Handstand", 1301, ""),
    ("CrossStep", 1302, ""),
    ("OnesidedStep", 1303, ""),
    ("Bound", 1304, ""),
    ("LeadFollow", 1045, ""),
    ("LeftFlip", 1042, ""),
    ("RightFlip", 1043, ""),
    ("Backflip", 1044, "")
]


class MyUnitreeSkills(AbstractSkill):
    """My Unitree Skills."""

    _robot: Optional[Robot] = None

    def __init__(self, robot: Optional[Robot] = None, **data):
        super().__init__(**data)
        self._robot: Robot = robot

        # Create the skills and add them to the list of skills
        self.add_skills(self.create_skills_live())
        nested_skills = self.get_nested_skills()
        self.set_list_of_skills(nested_skills)

        # Provide the robot instance to each skill
        for skill_class in nested_skills:
            print("\033[92mCreating instance for skill: {}\033[0m".format(
                skill_class))
            self.create_instance(skill_class.__name__, robot=robot)

    def create_skills_live(self) -> List[AbstractSkill]:
        # ================================================
        # Procedurally created skills
        # ================================================
        class BaseUnitreeSkill(AbstractSkill):
            """Base skill for dynamic skill creation."""
            _robot: Optional[Robot] = None

            def __init__(self, robot: Optional[Robot] = None, **data):
                super().__init__(**data)
                self._robot = robot

            def __call__(self):
                _GREEN_PRINT_COLOR = "\033[32m"
                _RESET_COLOR = "\033[0m"
                string = f"{_GREEN_PRINT_COLOR}This is a base skill, created for the specific skill: {self._app_id}{_RESET_COLOR}"
                print(string)
                if self._robot is None:
                    raise RuntimeError(
                        "No Robot instance provided to {self.__class__.__name__} Skill"
                    )
                elif self._app_id is None:
                    raise RuntimeError(
                        "No App ID provided to {self.__class__.__name__} Skill")
                else:
                    self._robot.webrtc_req(api_id=self._app_id)
                    string = f"{_GREEN_PRINT_COLOR}{self.__class__.__name__} was successful: id={self._app_id}{_RESET_COLOR}"
                    print(string)
                    return string

        skills_classes = []
        for name, app_id, description in UNITREE_ROS_CONTROLS:
            skill_class = type(
                name,  # Name of the class
                (BaseUnitreeSkill,),  # Base classes
                {
                    '__doc__': description,
                    '_app_id': app_id
                })
            skills_classes.append(skill_class)

        return skills_classes

    class Move(AbstractSkill):
        """Move the robot forward using distance commands."""

        _robot: Robot = None
        _GREEN_PRINT_COLOR: str = "\033[32m"
        _RESET_COLOR: str = "\033[0m"

        distance: float = Field(..., description="Distance to move in meters")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(
                f"{self._GREEN_PRINT_COLOR}Initializing Move Skill{self._RESET_COLOR}"
            )
            self._robot = robot
            print(
                f"{self._GREEN_PRINT_COLOR}Move Skill Initialized with Robot: {self._robot}{self._RESET_COLOR}"
            )

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Move Skill")
            elif self._robot.ros_control is None:
                raise RuntimeError(
                    "No ROS control interface available for movement")
            else:
                return self._robot.ros_control.move(distance=self.distance)

    class Reverse(AbstractSkill):
        """Reverse the robot using distance commands."""

        _robot: Robot = None
        _GREEN_PRINT_COLOR: str = "\033[32m"
        _RESET_COLOR: str = "\033[0m"

        distance: float = Field(...,
                                description="Distance to reverse in meters")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(
                f"{self._GREEN_PRINT_COLOR}Initializing Reverse Skill{self._RESET_COLOR}"
            )
            self._robot = robot
            print(
                f"{self._GREEN_PRINT_COLOR}Reverse Skill Initialized with Robot: {self._robot}{self._RESET_COLOR}"
            )

        def __call__(self):
            if self._robot is None:
                raise RuntimeError(
                    "No Robot instance provided to Reverse Skill")
            elif self._robot.ros_control is None:
                raise RuntimeError(
                    "No ROS control interface available for movement")
            else:
                return self._robot.ros_control.reverse(distance=self.distance)

    class SpinLeft(AbstractSkill):
        """Spin the robot left using degree commands."""

        _robot: Robot = None
        _GREEN_PRINT_COLOR: str = "\033[32m"
        _RESET_COLOR: str = "\033[0m"

        degrees: float = Field(...,
                               description="Distance to spin left in degrees")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(
                f"{self._GREEN_PRINT_COLOR}Initializing SpinLeft Skill{self._RESET_COLOR}"
            )
            self._robot = robot
            print(
                f"{self._GREEN_PRINT_COLOR}SpinLeft Skill Initialized with Robot: {self._robot}{self._RESET_COLOR}"
            )

        def __call__(self):
            if self._robot is None:
                raise RuntimeError(
                    "No Robot instance provided to SpinLeft Skill")
            elif self._robot.ros_control is None:
                raise RuntimeError(
                    "No ROS control interface available for movement")
            else:
                return self._robot.ros_control.spin(
                    degrees=self.degrees)  # Spinning left is positive degrees

    class SpinRight(AbstractSkill):
        """Spin the robot right using degree commands."""

        _robot: Robot = None
        _GREEN_PRINT_COLOR: str = "\033[32m"
        _RESET_COLOR: str = "\033[0m"

        degrees: float = Field(...,
                               description="Distance to spin right in degrees")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(
                f"{self._GREEN_PRINT_COLOR}Initializing SpinRight Skill{self._RESET_COLOR}"
            )
            self._robot = robot
            print(
                f"{self._GREEN_PRINT_COLOR}SpinRight Skill Initialized with Robot: {self._robot}{self._RESET_COLOR}"
            )

        def __call__(self):
            if self._robot is None:
                raise RuntimeError(
                    "No Robot instance provided to SpinRight Skill")
            elif self._robot.ros_control is None:
                raise RuntimeError(
                    "No ROS control interface available for movement")
            else:
                return self._robot.ros_control.spin(
                    degrees=-self.degrees)  # Spinning right is negative degrees

    class Wait(AbstractSkill):
        """Wait for a specified amount of time."""

        _robot: Robot = None
        _GREEN_PRINT_COLOR: str = "\033[32m"
        _RESET_COLOR: str = "\033[0m"

        seconds: float = Field(..., description="Seconds to wait")

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            print(
                f"{self._GREEN_PRINT_COLOR}Initializing Wait Skill{self._RESET_COLOR}"
            )
            self._robot = robot
            print(
                f"{self._GREEN_PRINT_COLOR}Wait Skill Initialized with Robot: {self._robot}{self._RESET_COLOR}"
            )

        def __call__(self):
            if self._robot is None:
                raise RuntimeError(
                    "No Robot instance provided to SpinRight Skill")
            elif self._robot.ros_control is None:
                raise RuntimeError(
                    "No ROS control interface available for movement")
            else:
                return time.sleep(self.seconds)

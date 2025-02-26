
from typing import List, Optional, Type

from pydantic import Field
from dimos.robot.robot import Robot
from dimos.robot.skills import AbstractSkill

@staticmethod
def get_my_ros_controls_app_ids():
    return [
            ("Damp", 1001),
            ("BalanceStand", 1002),
            ("StopMove", 1003),
            ("StandUp", 1004),
            ("StandDown", 1005),
            ("RecoveryStand", 1006),
            ("Euler", 1007),
            # ("Move", 1008),
            ("Sit", 1009),
            ("RiseSit", 1010),
            ("SwitchGait", 1011),
            ("Trigger", 1012),
            ("BodyHeight", 1013),
            ("FootRaiseHeight", 1014),
            ("SpeedLevel", 1015),
            ("Hello", 1016),
            ("Stretch", 1017),
            ("TrajectoryFollow", 1018),
            ("ContinuousGait", 1019),
            ("Content", 1020),
            ("Wallow", 1021),
            ("Dance1", 1022),
            ("Dance2", 1023),
            ("GetBodyHeight", 1024),
            ("GetFootRaiseHeight", 1025),
            ("GetSpeedLevel", 1026),
            ("SwitchJoystick", 1027),
            ("Pose", 1028),
            ("Scrape", 1029),
            ("FrontFlip", 1030),
            ("FrontJump", 1031),
            ("FrontPounce", 1032),
            ("WiggleHips", 1033),
            ("GetState", 1034),
            ("EconomicGait", 1035),
            ("FingerHeart", 1036),
            ("Handstand", 1301),
            ("CrossStep", 1302),
            ("OnesidedStep", 1303),
            ("Bound", 1304),
            ("LeadFollow", 1045),
            ("LeftFlip", 1042),
            ("RightFlip", 1043),
            ("Backflip", 1044)
        ]

class MyUnitreeSkills(AbstractSkill):
    """My Unitree Skills."""

    _robot: Optional[Robot] = None
    _nested_skills: List[AbstractSkill] = []

    def __init__(self, robot: Optional[Robot] = None, **data):
        super().__init__(**data)
        self._robot: Robot = robot
        self._nested_skills = self.create_skills_live()

    def create_skills_live(self) -> List[AbstractSkill]:
        class BaseSkill(AbstractSkill):
            """Base skill for dynamic skill creation."""
            _robot: Optional[Robot] = None

            def __init__(self, robot: Optional[Robot] = None, **data):
                super().__init__(**data)
                self._robot = robot

            def __call__(self):
                _MOVE_PRINT_COLOR = "\033[32m"
                _MOVE_RESET_COLOR = "\033[0m"  
                raise print(f"{_MOVE_PRINT_COLOR}This is a base skill, created for the specific skill: {self._app_id}{_MOVE_RESET_COLOR}")

        skills_classes = []
        for name, app_id in get_my_ros_controls_app_ids():
            skill_class = type(
                name,  # Name of the class
                (BaseSkill,),  # Base classes
                {
                    '__doc__': f'Automatically generated skill class for {name}',
                    '_app_id': app_id
                }
            )
            skills_classes.append(skill_class)
        
        return skills_classes


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
        
    # ================================================
    # Procedurally created nested skills
    # ================================================
    class Damp(AbstractSkill):
        """Reduces the impact of vibrations and shocks on the robot's hardware during movement."""
        _robot: Optional[Robot] = None
        _app_id = 1001

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Damp Skill")
            else:
                return f"Damp was successful: id={self._app_id}"

    class BalanceStand(AbstractSkill):
        """Activates a mode that maintains the robot in a balanced standing position, adjusting dynamically to slight movements or slopes."""
        _robot: Optional[Robot] = None
        _app_id = 1002

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to BalanceStand Skill")
            else:
                return f"BalanceStand was successful: id={self._app_id}"

    class StopMove(AbstractSkill):
        """Immediately stops all ongoing movement commands to the robot, bringing it to a stationary position."""
        _robot: Optional[Robot] = None
        _app_id = 1003

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to StopMove Skill")
            else:
                return f"StopMove was successful: id={self._app_id}"

    class StandUp(AbstractSkill):
        """Commands the robot to transition from a sitting or prone position to a standing posture."""
        _robot: Optional[Robot] = None
        _app_id = 1004

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to StandUp Skill")
            else:
                return f"StandUp was successful: id={self._app_id}"

    class StandDown(AbstractSkill):
        """Instructs the robot to move from a standing position to a sitting or prone posture."""
        _robot: Optional[Robot] = None
        _app_id = 1005

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to StandDown Skill")
            else:
                return f"StandDown was successful: id={self._app_id}"

    class RecoveryStand(AbstractSkill):
        """Engages a sequence to recover the robot to a standing position if it has fallen or been displaced."""
        _robot: Optional[Robot] = None
        _app_id = 1006

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to RecoveryStand Skill")
            else:
                return f"RecoveryStand was successful: id={self._app_id}"

    class Euler(AbstractSkill):
        """Adjusts the robot’s orientation using Euler angles, providing precise control over its rotation."""
        _robot: Optional[Robot] = None
        _app_id = 1007

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Euler Skill")
            else:
                return f"Euler was successful: id={self._app_id}"

    class Sit(AbstractSkill):
        """Commands the robot to sit down from a standing or moving stance."""
        _robot: Optional[Robot] = None
        _app_id = 1009

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Sit Skill")
            else:
                return f"Sit was successful: id={self._app_id}"

    class RiseSit(AbstractSkill):
        """Commands the robot to rise back to a standing position from a sitting posture."""
        _robot: Optional[Robot] = None
        _app_id = 1010

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to RiseSit Skill")
            else:
                return f"RiseSit was successful: id={self._app_id}"

    class SwitchGait(AbstractSkill):
        """Switches the robot's walking pattern or style dynamically, suitable for different terrains or speeds."""
        _robot: Optional[Robot] = None
        _app_id = 1011

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to SwitchGait Skill")
            else:
                return f"SwitchGait was successful: id={self._app_id}"

    class Trigger(AbstractSkill):
        """Triggers a specific action or custom routine programmed into the robot."""
        _robot: Optional[Robot] = None
        _app_id = 1012

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Trigger Skill")
            else:
                return f"Trigger was successful: id={self._app_id}"

    class BodyHeight(AbstractSkill):
        """Adjusts the height of the robot’s body from the ground, useful for navigating various obstacles."""
        _robot: Optional[Robot] = None
        _app_id = 1013

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to BodyHeight Skill")
            else:
                return f"BodyHeight was successful: id={self._app_id}"

    class FootRaiseHeight(AbstractSkill):
        """Controls how high the robot lifts its feet during movement, which can be adjusted for different surfaces."""
        _robot: Optional[Robot] = None
        _app_id = 1014

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to FootRaiseHeight Skill")
            else:
                return f"FootRaiseHeight was successful: id={self._app_id}"

    class SpeedLevel(AbstractSkill):
        """Sets or adjusts the speed at which the robot moves, with various levels available for different operational needs."""
        _robot: Optional[Robot] = None
        _app_id = 1015

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to SpeedLevel Skill")
            else:
                return f"SpeedLevel was successful: id={self._app_id}"

    class Hello(AbstractSkill):
        """Performs a greeting action, which could involve a wave or other friendly gesture."""
        _robot: Optional[Robot] = None
        _app_id = 1016

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Hello Skill")
            else:
                return f"Hello was successful: id={self._app_id}"

    class Stretch(AbstractSkill):
        """Engages the robot in a stretching routine to maintain or enhance mechanical flexibility."""
        _robot: Optional[Robot] = None
        _app_id = 1017

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Stretch Skill")
            else:
                return f"Stretch was successful: id={self._app_id}"

    class TrajectoryFollow(AbstractSkill):
        """Directs the robot to follow a predefined trajectory, which could involve complex paths or maneuvers."""
        _robot: Optional[Robot] = None
        _app_id = 1018

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to TrajectoryFollow Skill")
            else:
                return f"TrajectoryFollow was successful: id={self._app_id}"

    class ContinuousGait(AbstractSkill):
        """Enables a mode for continuous walking or running, ideal for long-distance travel."""
        _robot: Optional[Robot] = None
        _app_id = 1019

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to ContinuousGait Skill")
            else:
                return f"ContinuousGait was successful: id={self._app_id}"

    class Content(AbstractSkill):
        """Displays or triggers content, which could involve audio-visual outputs or interactions."""
        _robot: Optional[Robot] = None
        _app_id = 1020

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Content Skill")
            else:
                return f"Content was successful: id={self._app_id}"

    class Wallow(AbstractSkill):
        """Engages a playful or restful activity, possibly mimicking animal-like wallowing."""
        _robot: Optional[Robot] = None
        _app_id = 1021

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Wallow Skill")
            else:
                return f"Wallow was successful: id={self._app_id}"

    class Dance1(AbstractSkill):
        """Performs a predefined dance routine 1, programmed for entertainment or demonstration."""
        _robot: Optional[Robot] = None
        _app_id = 1022

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Dance1 Skill")
            else:
                return f"Dance1 was successful: id={self._app_id}"

    class Dance2(AbstractSkill):
        """Performs another variant of a predefined dance routine 2."""
        _robot: Optional[Robot] = None
        _app_id = 1023

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Dance2 Skill")
            else:
                return f"Dance2 was successful: id={self._app_id}"

    class GetBodyHeight(AbstractSkill):
        """Retrieves the current height of the robot's body from the ground."""
        _robot: Optional[Robot] = None
        _app_id = 1024

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to GetBodyHeight Skill")
            else:
                return f"GetBodyHeight was successful: id={self._app_id}"

    class GetFootRaiseHeight(AbstractSkill):
        """Retrieves the current height at which the robot’s feet are being raised during movement."""
        _robot: Optional[Robot] = None
        _app_id = 1025

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to GetFootRaiseHeight Skill")
            else:
                return f"GetFootRaiseHeight was successful: id={self._app_id}"

    class GetSpeedLevel(AbstractSkill):
        """Returns the current speed level at which the robot is operating."""
        _robot: Optional[Robot] = None
        _app_id = 1026

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to GetSpeedLevel Skill")
            else:
                return f"GetSpeedLevel was successful: id={self._app_id}"

    class SwitchJoystick(AbstractSkill):
        """Toggles the control mode to joystick input, allowing for manual direction of the robot’s movements."""
        _robot: Optional[Robot] = None
        _app_id = 1027

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to SwitchJoystick Skill")
            else:
                return f"SwitchJoystick was successful: id={self._app_id}"

    class Pose(AbstractSkill):
        """Directs the robot to take a specific pose or stance, which could be used for tasks or performances."""
        _robot: Optional[Robot] = None
        _app_id = 1028

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Pose Skill")
            else:
                return f"Pose was successful: id={self._app_id}"

    class Scrape(AbstractSkill):
        """Engages a scraping motion, possibly for clearing debris or interacting with the environment."""
        _robot: Optional[Robot] = None
        _app_id = 1029

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Scrape Skill")
            else:
                return f"Scrape was successful: id={self._app_id}"

    class FrontFlip(AbstractSkill):
        """Executes a front flip, a complex acrobatic maneuver for demonstration or obstacle navigation."""
        _robot: Optional[Robot] = None
        _app_id = 1030

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to FrontFlip Skill")
            else:
                return f"FrontFlip was successful: id={self._app_id}"

    class FrontJump(AbstractSkill):
        """Commands the robot to perform a forward jump, useful for clearing obstacles or for show."""
        _robot: Optional[Robot] = None
        _app_id = 1031

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to FrontJump Skill")
            else:
                return f"FrontJump was successful: id={self._app_id}"

    class FrontPounce(AbstractSkill):
        """Initiates a pouncing movement forward, mimicking animal-like pouncing behavior."""
        _robot: Optional[Robot] = None
        _app_id = 1032

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to FrontPounce Skill")
            else:
                return f"FrontPounce was successful: id={self._app_id}"

    class WiggleHips(AbstractSkill):
        """Causes the robot to wiggle its hips, likely for a playful effect or to demonstrate agility."""
        _robot: Optional[Robot] = None
        _app_id = 1033

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to WiggleHips Skill")
            else:
                return f"WiggleHips was successful: id={self._app_id}"

    class GetState(AbstractSkill):
        """Retrieves the current operational state of the robot, including status reports or diagnostic information."""
        _robot: Optional[Robot] = None
        _app_id = 1034

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to GetState Skill")
            else:
                return f"GetState was successful: id={self._app_id}"

    class EconomicGait(AbstractSkill):
        """Engages a more energy-efficient walking or running mode to conserve battery life."""
        _robot: Optional[Robot] = None
        _app_id = 1035

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to EconomicGait Skill")
            else:
                return f"EconomicGait was successful: id={self._app_id}"

    class FingerHeart(AbstractSkill):
        """Performs a finger heart gesture, popular in some cultures as a friendly or loving gesture."""
        _robot: Optional[Robot] = None
        _app_id = 1036

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to FingerHeart Skill")
            else:
                return f"FingerHeart was successful: id={self._app_id}"

    class Handstand(AbstractSkill):
        """Commands the robot to perform a handstand, demonstrating balance and control."""
        _robot: Optional[Robot] = None
        _app_id = 1301

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Handstand Skill")
            else:
                return f"Handstand was successful: id={self._app_id}"

    class CrossStep(AbstractSkill):
        """Engages the robot in a cross-stepping routine, useful for complex locomotion or dance moves."""
        _robot: Optional[Robot] = None
        _app_id = 1302

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to CrossStep Skill")
            else:
                return f"CrossStep was successful: id={self._app_id}"

    class OnesidedStep(AbstractSkill):
        """Commands the robot to perform a stepping motion that predominantly uses one side."""
        _robot: Optional[Robot] = None
        _app_id = 1303

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to OnesidedStep Skill")
            else:
                return f"OnesidedStep was successful: id={self._app_id}"

    class Bound(AbstractSkill):
        """Initiates a bounding motion, similar to a light, repetitive hopping or leaping."""
        _robot: Optional[Robot] = None
        _app_id = 1304

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Bound Skill")
            else:
                return f"Bound was successful: id={self._app_id}"

    class LeadFollow(AbstractSkill):
        """Engages follow-the-leader behavior, where the robot follows a designated leader or follows a signal."""
        _robot: Optional[Robot] = None
        _app_id = 1045

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to LeadFollow Skill")
            else:
                return f"LeadFollow was successful: id={self._app_id}"

    class LeftFlip(AbstractSkill):
        """Executes a flip towards the left side, combining acrobatics with directional control."""
        _robot: Optional[Robot] = None
        _app_id = 1042

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to LeftFlip Skill")
            else:
                return f"LeftFlip was successful: id={self._app_id}"

    class RightFlip(AbstractSkill):
        """Performs a flip towards the right side, adding an acrobatic element to its repertoire."""
        _robot: Optional[Robot] = None
        _app_id = 1043

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to RightFlip Skill")
            else:
                return f"RightFlip was successful: id={self._app_id}"

    class Backflip(AbstractSkill):
        """Executes a backflip, a highly skilled maneuver for showing agility and control."""
        _robot: Optional[Robot] = None
        _app_id = 1044

        def __init__(self, robot: Optional[Robot] = None, **data):
            super().__init__(**data)
            self._robot = robot

        def __call__(self):
            if self._robot is None:
                raise RuntimeError("No Robot instance provided to Backflip Skill")
            else:
                return f"Backflip was successful: id={self._app_id}"

import asyncio
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from openai import pydantic_function_tool
import requests

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SkillRegistry:
    def __init__(self):
        self.skills = {}

    def register_skill(self, name, skill):
        self.skills[name] = skill


class AbstractSkill(BaseModel):

    _instances: dict[str, dict] = {}

    def __init__(self, *args, **kwargs):
        print("Initializing AbstractSkill Class")
        super().__init__(*args, **kwargs)
        self._instances = {}
        print(f"Instances: {self._instances}")
    
    def create_instance(self, name, args):
        # Key based only on the name
        key = name
        
        print(f"Preparing to create instance with name: {name} and args: {args}")

        if key not in self._instances:
            # Instead of creating an instance, store the args for later use
            self._instances[key] = args
            print(f"Stored args for later instance creation: {name} with args: {args}")

    def call_function(self, name, **args):
        # Get the stored args if available; otherwise, use an empty dict
        stored_args = self._instances.get(name, {})

        # Merge the arguments with priority given to stored arguments
        complete_args = {**args, **stored_args}

        try:
            # Dynamically get the class from the module or current script
            skill_class = getattr(self, name, None)
            if skill_class is None:
                raise ValueError(f"Skill class not found: {name}")

            # Initialize the instance with the merged arguments
            instance = skill_class(**complete_args)
            print(f"Instance created and function called for: {name} with args: {complete_args}")
            # Assuming the instance can be called directly
            return instance()
        except Exception as e:
            print(f"Error running function {name}: {e}")
            return f"Error running function {name}: {e}"

    # ==== Tools ====
    def set_tools(self, tools: Any):
        self._tools = tools

    def get_tools(self) -> Any:
        return self._tools


class SkillsHelper:
    @staticmethod
    def get_skill_as_json(skill: AbstractSkill) -> str:
        return pydantic_function_tool(skill)

    @staticmethod
    def get_nested_skills(skill: AbstractSkill) -> list[AbstractSkill]:
        nested_skills = []
        for attr_name in dir(skill):
            attr = getattr(skill, attr_name)
            if isinstance(attr, type) and issubclass(attr, AbstractSkill) and attr is not AbstractSkill:
                nested_skills.append(attr)
        return nested_skills

    @staticmethod
    def get_nested_skills_as_json(skill: AbstractSkill) -> list[str]:
        nested_skills = SkillsHelper.get_nested_skills(skill)
        nested_skills_json = list(map(pydantic_function_tool, nested_skills))
        return nested_skills_json
    
    @staticmethod
    def get_list_of_skills_as_json(list_of_skills: list[AbstractSkill]) -> list[str]:
        return list(map(pydantic_function_tool, list_of_skills))


# class Skills(AbstractSkill):
    
#     # This field will be excluded from the schema
#     _robot_video_provider: Optional[UnitreeVideoProvider] = None
#     _robot: Optional[Robot] = None

#     def __init__(self, robot_video_provider: Optional[UnitreeVideoProvider] = None, robot: Optional[Robot] = None, **data):
#         print("Initializing Skills Class")
#         super().__init__(**data)
#         self._robot_video_provider = robot_video_provider # TODO: Remove WebRTC
#         self._robot = robot

#     class MoveX(AbstractSkill):
#         """Moves the robot's arm along the X-axis.

#         Args:
#             distance (float): The distance to move along the X-axis in Units.

#         Returns:
#             str: A message indicating the movement.
#         """

#         distance: float = Field(...)

#         def __call__(self):
#             logger.info(f"Moving along X-axis by {self.distance} units.")
#             return f"Moved along X-axis by {self.distance} units."

#     class MoveY(AbstractSkill):
#         """Moves the robot's arm along the Y-axis.

#         Args:
#             distance (float): The distance to move along the Y-axis in Units.

#         Returns:
#             str: A message indicating the movement.
#         """

#         distance: float = Field(...)

#         def __call__(self):
#             logger.info(f"Moving along Y-axis by {self.distance} units.")
#             return f"Moved along Y-axis by {self.distance} units."

#     class GripArm(AbstractSkill):
#         """Activates the robotic gripper.

#         Returns:
#             str: A message indicating the gripper action.
#         """

#         class AnotherAnotherSkill(AbstractSkill):
#             """
#             This is a another test skill
#             """
#             pass

#         def __call__(self):
#             logger.info("Gripping with robotic arm.")
#             return "Gripped with robotic arm."

#     class GetWeather(AbstractSkill):
#         """
#         Retrieve the current temperature in Fahrenheit for a specified location.

#         Attributes:
#             latitude (str): Latitude of the location (e.g., "37.773972" for San Francisco, CA).
#             longitude (str): Longitude of the location (e.g., "-122.431297" for San Francisco, CA).

#         Returns:
#             str: The current temperature in Fahrenheit.
#         """
#         latitude: str = Field(...)
#         longitude: str = Field(...)

#         def __call__(self):
#             logger.info(f"Getting weather for {self.latitude}, {self.longitude}.")
#             response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={self.latitude}&longitude={self.longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&temperature_unit=fahrenheit")
#             data = response.json()
#             return data['current']['temperature_2m']
 

#     class Wiggle(AbstractSkill):
#         """Wiggles the robot's hips."""

#         _robot_video_provider: UnitreeVideoProvider = None
#         _WIGGLE_PRINT_COLOR: str = "\033[32m"
#         _WIGGLE_RESET_COLOR: str = "\033[0m"

#         # This field will be excluded from the schema
#         def __init__(self, robot_video_provider: Optional[UnitreeVideoProvider] = None, **data):
#             super().__init__(**data)
#             print(f"{self._WIGGLE_PRINT_COLOR}Initializing Wiggle Skill{self._WIGGLE_RESET_COLOR}")
#             self._robot_video_provider = robot_video_provider
#             print(f"{self._WIGGLE_PRINT_COLOR}Wiggle Skill Initialized with Robot Video Provider: {self._robot_video_provider}{self._WIGGLE_RESET_COLOR}")

#         def __call__(self):
#             #_WIGGLE_PRINT_COLOR: str = "\033[32m"
#             #_WIGGLE_RESET_COLOR: str = "\033[0m"
#             print(f"{self._WIGGLE_PRINT_COLOR}Trying to Wiggle the robot's hips.{self._WIGGLE_RESET_COLOR}")
#             try:
#                 # self._robot_video_provider.conn.datachannel.sendStandUp()
#                 self._robot_video_provider.conn.datachannel.sendWiggle()
#                 return f"{self._WIGGLE_PRINT_COLOR}Wiggled the robot's hips.{self._WIGGLE_RESET_COLOR}"
#             except Exception as e:
#                 return f"{self._WIGGLE_PRINT_COLOR}Error wiggling the robot's hips: {e}{self._WIGGLE_RESET_COLOR}"

# Singleton instantiation will come at the robot class
# Possibly two classes: Robot skills abstract & reasoning skills / perception skills
# Reach skill will differ from object recognition skill (may need separate classes)
# NavStack.export to skill func as well as other skills will use the same export function / helper

    # class WiggleContainer():

    #     # Parameters from initialization
    #     def __init__(self, robot_video_provider: UnitreeVideoProvider):
    #         self.robot_video_provider = robot_video_provider

    #     # Function to execute the skill
    #     async def execute(self):
    #         logger.info("Wiggling the robot's hips.")
    #         try:
    #             await self.robot_video_provider.conn.datachannel.sendWiggle()
    #             await asyncio.sleep(3)
    #             return "Wiggled the robot's hips."
    #         except Exception as e:
    #             return f"Error wiggling the robot's hips: {e}"

    #     class Wiggle(AbstractSkill):
    #         """Wiggles the robot's hips."""

    #     pydantic_class: AbstractSkill = Wiggle()

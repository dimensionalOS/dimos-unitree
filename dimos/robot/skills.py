import logging
from typing import Any
from pydantic import BaseModel

from openai import pydantic_function_tool

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SkillRegistry:
    def __init__(self):
        self.skills = [AbstractSkill]

    def register_skill(self, skill: "AbstractSkill"):
        self.skills.append(skill)

    def get_skills(self) -> list["AbstractSkill"]:
        return self.skills


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
            
            # Call the instance directly
            return instance()
        except Exception as e:
            print(f"Error running function {name}: {e}")
            return f"Error running function {name}: {e}"

    def clone(self) -> "AbstractSkill":
        return AbstractSkill()

    # ==== Tools ====
    def set_list_of_skills(self, list_of_skills: list["AbstractSkill"]):
        self._list_of_skills = list_of_skills

    def get_tools(self) -> Any:
        return SkillsHelper.get_list_of_skills_as_json(list_of_skills=self._list_of_skills)


class SkillsHelper:
    @staticmethod
    def get_skill_as_json(skill: AbstractSkill) -> str:
        return pydantic_function_tool(skill)

    @staticmethod
    def get_nested_skills(skill: AbstractSkill) -> list[AbstractSkill]:
        nested_skills = []
        for attr_name in dir(skill):
            # Skip dunder attributes that cause issues
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(skill, attr_name)
            except AttributeError:
                continue
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


# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tests.test_header

from dimos.skills.skills import AbstractSkill, SkillRegistry
from dimos.robot.robot import MockRobot
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills
from dimos.types.constants import Colors
from dimos.agents.agent import OpenAIAgent

class TestSkillRegistry:
    robot = MockRobot()
    skill_group = MyUnitreeSkills(robot=robot)

    # Print out all available skills
    print("\nAvailable Unitree Robot Skills:")
    for skill in skill_group.collect_skills():
        print(f"- {skill.__name__}")

    print(f"\n{Colors.RED_PRINT_COLOR}Get the skills{Colors.RESET_COLOR}")
    for skill in skill_group.collect_skills():
        if skill.__name__ == "HelloAndStuff":
            print(f"{Colors.GREEN_PRINT_COLOR}Calling skill: {skill.__name__}{Colors.RESET_COLOR}")
            skill()
            print("Done.")

    # Initialize the skills
    print(f"\n{Colors.RED_PRINT_COLOR}Initialize the skills{Colors.RESET_COLOR}")
    skill_group.initialize_skills()

    # Add the skills to the skill registry
    print(f"\n{Colors.RED_PRINT_COLOR}Add the skills to the skill registry{Colors.RESET_COLOR}")
    for skill in skill_group.collect_skills():
        skill_group.add_to_skill_registry(skill)

    # Call the skills
    for skill in skill_group.collect_skills():
        if skill.__name__ == "HelloAndStuff":
            print(f"{Colors.GREEN_PRINT_COLOR}Calling skill: {skill.__name__}{Colors.RESET_COLOR}")
            skill_group.skill_registry.call_function(skill.__name__)
            print("Done.")

class TestSkillWithAgent:
    def __init__(self):
        # Create a skill registry and initialize with mock robot
        self.skill_registry = SkillRegistry()
        self.robot = MockRobot()
        self.skill_group = MyUnitreeSkills(robot=self.robot)
        
        # Initialize the skills
        print(f"\n{Colors.BLUE_PRINT_COLOR}Initializing skills for agent test{Colors.RESET_COLOR}")
        self.skill_group.initialize_skills()
        
        # Add a new skill to the registry
        class TestSkill(AbstractSkill):
            """A test skill that prints a message."""

            def __call__(self):
                print("Some sample skill was called.")

        self.skill_group.add_to_skill_registry(TestSkill)
        for skill in self.skill_group.collect_skills():
            self.skill_group.add_to_skill_registry(skill)
            print(f"- Registered: {skill.__name__}")
        
        # Create an OpenAIAgent with the skills
        print(f"\n{Colors.BLUE_PRINT_COLOR}Creating agent with skills{Colors.RESET_COLOR}")
        self.agent = OpenAIAgent(
            dev_name="SkillTestAgent",
            system_query="You are a skill testing agent. When prompted to perform an action, use the appropriate skill.",
            skills=self.skill_group
        )
        
        # Test the agent with a query that should trigger a skill
        print(f"\n{Colors.BLUE_PRINT_COLOR}Testing agent with skills{Colors.RESET_COLOR}")
        response = self.agent.run_observable_query("Please say hello").run()
        print(f"{Colors.GREEN_PRINT_COLOR}Agent response: {response}{Colors.RESET_COLOR}")
        
        # Test skill execution through agent
        print(f"\n{Colors.BLUE_PRINT_COLOR}Testing skill execution through agent{Colors.RESET_COLOR}")
        response = self.agent.run_observable_query("Execute the HelloAndStuff skill").run()
        print(f"{Colors.GREEN_PRINT_COLOR}Agent response: {response}{Colors.RESET_COLOR}")

        # Test manual skill execution through agent
        print(f"\n{Colors.BLUE_PRINT_COLOR}Testing manual skill execution through agent{Colors.RESET_COLOR}")
        response = self.agent.run_observable_query("Execute the TestSkill skill").run()
        print(f"{Colors.GREEN_PRINT_COLOR}Agent response: {response}{Colors.RESET_COLOR}")
        
        print(f"\n{Colors.GREEN_PRINT_COLOR}Skill with agent test completed{Colors.RESET_COLOR}")

if __name__ == "__main__":
    print(f"{Colors.YELLOW_PRINT_COLOR}Running TestSkillRegistry{Colors.RESET_COLOR}")
    TestSkillRegistry()
    
    print(f"\n{Colors.YELLOW_PRINT_COLOR}Running TestSkillWithAgent{Colors.RESET_COLOR}")
    TestSkillWithAgent()

import sys
import os
import time

from dimos.robot.skills import SkillRegistry, SkillsHelper

# Add the parent directory of 'tests' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"Hi from {os.path.basename(__file__)}\n")

# -----

from dimos.agents.agent import OpenAIAgent
from dimos.robot.unitree.unitree_go2 import UnitreeGo2

# TODO: Cleanup Skills
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills

class UnitreeAgentDemo:
    def __init__(self):
        self.robot_ip = None
        self.connection_method = None
        self.serial_number = None
        self.output_dir = None
        self.api_call_interval = None
        self._fetch_env_vars()
        self._initialize_robot()

    def _fetch_env_vars(self):
        print("Fetching environment variables")

        def get_env_var(var_name, default=None, required=False):
            """Get environment variable with validation."""
            value = os.getenv(var_name, default)
            if required and not value:
                raise ValueError(f"{var_name} environment variable is required")
            return value

        self.robot_ip = get_env_var("ROBOT_IP", required=True)
        self.connection_method = get_env_var("CONN_TYPE")
        self.serial_number = get_env_var("SERIAL_NUMBER")
        self.output_dir = get_env_var("ROS_OUTPUT_DIR", os.path.join(os.getcwd(), "assets/output/ros"))
        self.api_call_interval = get_env_var("API_CALL_INTERVAL", "5")

    def _initialize_robot(self):
        print("Initializing Unitree Robot")
        self.robot = UnitreeGo2(
            ip=self.robot_ip,
            connection_method=self.connection_method,
            serial_number=self.serial_number,
            output_dir=self.output_dir,
            api_call_interval=self.api_call_interval
        )
    
    def run(self):
        # Whether to use the ros or test video stream
        def get_stream(use_ros: bool = True):
            print("Starting Unitree Perception Stream")
            if use_ros:
                return self.robot.get_ros_video_stream()
            else:
                from dimos.stream.video_provider import VideoProvider
                return VideoProvider(
                    dev_name="UnitreeGo2",
                    video_source="/app/assets/framecount.mp4"
                ).capture_video_as_observable()

        # Initialize video stream
        self.video_stream = get_stream(use_ros=True)

        # TODO: Cleanup Skills
        # def get_skills_instance():
        #     skills_instance = MyUnitreeSkills(robot=self.robot)

        #     # skill_registry = SkillRegistry() 
        #     # skill_registry.register_skill(skills_instance.Move)
        #     # skill_registry.register_skill(skills_instance.Wave)
        #     # skill_registry.register_skill(skills_instance.get_nested_skills())

        #     skills_instance.set_list_of_skills(SkillsHelper.get_nested_skills(skills_instance))
        #     #skills_instance.set_list_of_skills([skills_instance.Move, skills_instance.Wave])
        #     #skills_instance.create_instance("Move", {"robot": self.robot})
        #     #skills_instance.create_instance("Wave", {"robot": self.robot})
        #     #skills_instance.create_instance("Damp", {"robot": self.robot})
        #     skills_instance.create_instance("BalanceStand", {"robot": self.robot})
        #     #skills_instance.create_instance("StopMove", {"robot": self.robot})
            
        #     return skills_instance
        def get_skills_instance():
            skills_instance = MyUnitreeSkills(robot=self.robot)
            
            # Retrieve the nested skill classes from the skills_instance.
            nested_skills = SkillsHelper.get_nested_skills(skills_instance)
            skills_instance.set_list_of_skills(nested_skills)
            
            # Create the dynamic skill registry.
            # skill_registry = SkillRegistry()
            # for skill in skills_instance.create_skills_live():
            #     skill_registry.register_skill(skill)
            # nested_skills = skill_registry.get_skills()

            # Set the list of skills for the skills_instance.
            # skills_instance.set_list_of_skills(nested_skills)

            # Automatically call create_instance for every nested skill with the robot parameter.
            for skill_class in nested_skills:
                skills_instance.create_instance(skill_class.__name__, {"robot": self.robot})
            
            return skills_instance
        
        print("Starting Unitree Perception Agent")
        self.UnitreePerceptionAgent = OpenAIAgent(
            dev_name="UnitreePerceptionAgent", 
            agent_type="Perception", 
            input_video_stream=self.video_stream,
            output_dir=self.output_dir,
            # query="Based on the image, if you do not see a human, rotate the robot at 0.5 rad/s for 1.5 second. If you do see a human, rotate the robot at -1.0 rad/s for 3 seconds.",
            query="Denote the number you see in the image. Only provide the number, without any other text in your response. If the number is above 500, but lower than 1000, then rotate the robot at 0.5 rad/s for 1 second. Is the number above 1000, but lower than 2000, then wave the robot's hand for a random duration between 1 and 3 seconds. If the number is above 2000, then maintain the robot in a balanced standing position.",
            image_detail="high",
            skills=get_skills_instance(),
            # pool_scheduler=self.thread_pool_scheduler,
            # frame_processor=frame_processor,
        )

    def stop(self):
        print("Stopping Unitree Agent")
        self.robot.cleanup()

if __name__ == "__main__":
    myUnitreeAgentDemo = UnitreeAgentDemo()
    myUnitreeAgentDemo.run()

    # Keep the program running to allow the Unitree Agent Demo to operate continuously
    try:
        print("\nRunning Unitree Agent Demo (Press Ctrl+C to stop)...")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping Unitree Agent Demo")
        myUnitreeAgentDemo.stop()
    except Exception as e:
        print(f"Error in main loop: {e}")

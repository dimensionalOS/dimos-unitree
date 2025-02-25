import sys
import os
import time

# Add the parent directory of 'tests' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"Hi from {os.path.basename(__file__)}\n")

# -----

from dimos.agents.agent import OpenAIAgent
from dimos.robot.unitree.unitree_go2 import UnitreeGo2

# TODO: Cleanup Skills
from dimos.robot.robot import MyUnitreeSkills
from dimos.robot.skills import AbstractSkill, SkillsHelper

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
            api_call_interval=self.api_call_interval,
            use_ros=self.use_ros,
            use_webrtc=self.use_webrtc
        )
        

    def run(self):
        print("Starting Unitree Perception Stream")
        self.ros_perception_stream = self.robot.start_ros_perception()

        def get_skills_instance():
            skills_instance = MyUnitreeSkills(robot=self)
            list_of_skills: list[AbstractSkill] = [skills_instance.Move]
            list_of_skills_json = SkillsHelper.get_list_of_skills_as_json(list_of_skills)
            skills_instance.create_instance("Move", {"robot": self})
            print(f"skills_instance: {skills_instance}")
            print(f"list_of_skills_json: {list_of_skills_json}")
            skills_instance.set_tools(list_of_skills_json)

        print("Starting Unitree Perception Agent")
        self.UnitreePerceptionAgent = OpenAIAgent(
            dev_name="UnitreePerceptionAgent", 
            agent_type="Perception", 
            input_video_stream=self.ros_perception_stream,
            output_dir=self.output_dir,
            # query="Based on the image, if you see a human, rotate the robot at 0.5 rad/s for 1 second.",
            query="Denote the number you see in the image. Only provide the number, without any other text in your response. If the number is above 500, but lower than 1000, then rotate the robot at 0.5 rad/s for 1 second.",
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

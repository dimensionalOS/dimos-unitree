import os
import time
import threading
from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills
from dimos.web.robot_web_interface import RobotWebInterface
import pickle

# connects to a robot, saves costmap as a pickle


def main():
    print("Initializing Unitree Go2 robot with local planner visualization...")

    # Initialize the robot with ROS control and skills
    robot = UnitreeGo2(
        ip=os.getenv("ROBOT_IP"),
        ros_control=UnitreeROSControl(),
        skills=MyUnitreeSkills(),
    )

    def save_costmap():
        while True:
            # this is a bit dumb tbh, we should have a stream :/
            costmap = robot.ros_control.get_global_costmap()

            if costmap is not None:
                # pickle costmap
                path = os.path.join(os.path.dirname(__file__), "costmapMsg.pickle")
                print("Costmap", costmap, path)
                with open(path, "wb") as f:
                    pickle.dump(costmap, f)
            time.sleep(2)

    # Get the camera stream
    video_stream = robot.get_ros_video_stream()

    costmap_thread = None
    try:
        # Set up web interface with both streams
        streams = {"camera": video_stream}

        # Create and start the web interface
        web_interface = RobotWebInterface(port=5555, **streams)

        # Wait for initialization
        print("Waiting for camera and systems to initialize...")

        time.sleep(2)

        costmap_thread = threading.Thread(
            target=save_costmap,
            daemon=True,
        )
        costmap_thread.start()

        print("Robot streams running")
        print("Web interface available at http://localhost:5555")
        print("Press Ctrl+C to exit")

        # Start web server (blocking call)
        web_interface.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        print("Cleaning up...")
        robot.cleanup()
        print("Test completed")


if __name__ == "__main__":
    main()

import os
import time
import sys
from reactivex import Subject, operators as RxOps

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills
from dimos.web.robot_web_interface import RobotWebInterface
from dimos.utils.logging_config import logger


def main():
    # Hardcoded parameters
    timeout = 30.0  # Maximum time to follow a person (seconds)
    distance = 2.0  # Desired distance to maintain from target (meters)
    
    print("Initializing Unitree Go2 robot...")
    
    # Initialize the robot with ROS control and skills
    robot = UnitreeGo2(
        ip=os.getenv('ROBOT_IP'),
        ros_control=UnitreeROSControl(),
        skills=MyUnitreeSkills(),
        enable_visual_servoing=True,
    )

    tracking_stream = robot.person_tracking_stream
    viz_stream = tracking_stream.pipe(
        RxOps.share(),
        RxOps.map(lambda x: x["viz_frame"] if x is not None else None),
        RxOps.filter(lambda x: x is not None),
    )
    video_stream = robot.get_ros_video_stream()
    
    try:
        # Set up web interface
        logger.info("Initializing web interface")
        streams = {
            "unitree_video": video_stream,
            "person_tracking": viz_stream
        }
        
        web_interface = RobotWebInterface(
            port=5555,
            **streams
        )
        
        # Wait for camera and tracking to initialize
        print("Waiting for camera and tracking to initialize...")
        time.sleep(2)
        
        # Start following human in a separate thread
        import threading
        follow_thread = threading.Thread(
            target=lambda: robot.follow_human(timeout=timeout),
            daemon=True
        )
        follow_thread.start()
        
        print(f"Following human for {timeout} seconds...")
        print("Web interface available at http://localhost:5555")
        
        # Start web server (blocking call)
        web_interface.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        print("Test completed")
        robot.cleanup()


if __name__ == "__main__":
    main()

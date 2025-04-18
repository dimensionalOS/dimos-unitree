#!/usr/bin/env python3
import os
import time
import threading
import matplotlib.pyplot as plt
from reactivex import operators as ops
from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from costmap import Costmap
from draw import Drawer
from astar import astar
from path import Path
from vectortypes import VectorLike, Vector
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped


def transform_to_pos(msg: TransformStamped) -> [Vector, Vector]:
    q = msg.transform.rotation
    rotation = R.from_quat([q.x, q.y, q.z, q.w])
    return [
        Vector(msg.transform.translation).to_2d(),
        Vector(rotation.as_euler("zyx", degrees=False)),
    ]


def init_robot():
    print("Initializing Unitree Go2 robot with global planner visualization...")

    # Initialize the robot with ROS control and skills
    robot = UnitreeGo2(
        ip=os.getenv("ROBOT_IP"),
        ros_control=UnitreeROSControl(),
        # skills=MyUnitreeSkills(),
    )

    print("robot initialized")

    try:
        base_link = robot.ros_control.transform("base_link")
        position = base_link.pipe(ops.map(transform_to_euler))

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        print("Cleaning up...")
        robot.cleanup()
        print("Test completed")


def main():
    init_robot()


if __name__ == "__main__":
    main()

import matplotlib

# matplotlib.use("GTK3Cairo")
import os
import time
import threading
from geometry_msgs.msg import TransformStamped
import matplotlib.pyplot as plt

from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills
from dimos.web.robot_web_interface import RobotWebInterface
from costmap import Costmap
from draw import Drawer
from astar import astar
from path import Path
from vectortypes import VectorLike, Vector
from scipy.spatial.transform import Rotation as R

# connects to a robot, saves costmap as a pickle


def init_robot(env):
    print("Initializing Unitree Go2 robot with global planner visualization...")

    # Initialize the robot with ROS control and skills
    robot = UnitreeGo2(
        ip=os.getenv("ROBOT_IP"),
        ros_control=UnitreeROSControl(),
        # skills=MyUnitreeSkills(),
    )

    print("robot initialized")

    def sub_position():

        base_link = robot.ros_control.get_transform_stream(
            parent_frame="map", child_frame="base_link", frequency=20
        )

        def cb(msg):
            q = msg.transform.rotation
            rotation = R.from_quat([q.x, q.y, q.z, q.w])

            env.update({"base_link": msg})

            env["position"] = Vector(msg.transform.translation).to_2d()
            # euler angle, first value is yaw
            env["rotation"] = Vector(rotation.as_euler("zyx", degrees=False))

        base_link.subscribe(on_next=cb)

    def receive_costmap():
        while True:
            # this is a bit dumb tbh, we should have a stream :/
            costmap_msg = robot.ros_control.get_global_costmap()
            if costmap_msg is not None:
                env["costmap"] = Costmap.from_msg(costmap_msg).smudge(
                    preserve_unknown=True
                )
            time.sleep(1)

    sub_position()

    threading.Thread(
        target=receive_costmap,
        daemon=True,
    ).start()

    return robot


def main():

    env = {
        "target_orientation": 0,
        "costmap": Costmap.from_pickle("costmapMsg.pickle").smudge(),
        "position": Vector(0, 0),
        "destination": Vector(0, 0),
        "rotation": Vector(0, 0),
        "target": Vector(0, 0),
        "path": Path(),
        "is_navigating": False,
        "redraw": False,
    }

    robot = init_robot(env)

    def navigate_to(destination: VectorLike):
        print(f"Navigating to {destination}")
        env["destination"] = destination

        path = astar(env["costmap"], env["position"], destination, 50)

        if not path:
            print("Navigation fail")
            env["path"] = Path()
            return

        # Resample the path to have more evenly spaced points (easier for following)
        env["path"] = path.resample(0.5)  # 0.5m spacing between points

        # Set navigating flag
        env["is_navigating"] = True

        print(env["path"])

    def walk():
        position = env["position"]
        target = env["target"]

        # Calculate distance between current position and target
        distance = position.distance(target)

        # If we're more than 0.1m away from the target, move toward it
        if distance > 0.1:
            # Calculate direction vector from current position to target
            direction = (target - position).normalize()

            # Calculate velocities (linear x, y and angular z)
            x_vel = direction.x
            y_vel = direction.y
            yaw_vel = 0.0  # For simplicity, no rotation for now

            # Move the robot with 2 second duration
            robot.move_vel(x_vel, y_vel, yaw_vel, 2.0)

            print(f"Walking to target: {target}. Distance: {distance:.2f}m")
            return True
        else:
            print(f"Reached target: {target}")
            return False

    # Define helper functions for IPython
    def move(x, y, yaw=0.0, duration=2.0):
        """Move robot with velocity.

        Args:
            x: Forward/backward velocity (m/s)
            y: Left/right velocity (m/s)
            yaw: Rotational velocity (rad/s)
            duration: How long to move (seconds)
        """
        print(f"Moving with velocity: x={x}, y={y}, yaw={yaw}, duration={duration}")

        return robot.move_vel(x, y, yaw, duration)

    def pos():
        """Show current position."""
        position = env["position"]
        print(f"Current position: {position}")
        rotation = env["rotation"]
        print(f"Current rotation: {rotation}")
        return position

    def nav(x, y):
        """Navigate to position using A*."""
        destination = Vector(x, y)
        navigate_to(destination)
        return destination

    def auto():
        """Run automatic navigation mode."""
        if not env["is_navigating"] or len(env["path"]) == 0:
            print("No active navigation path. Use nav(x, y) first.")
            return False

        print("Starting automatic navigation...")
        try:
            while env["is_navigating"] and len(env["path"]) > 0:
                # Get the next target from the path
                env["target"] = env["path"].get_vector(0)

                # Try to walk to the target
                if not walk():
                    # If we reached the current target point, remove it from the path
                    if len(env["path"]) > 0:
                        print(f"Reached waypoint: {env['path'][0]}")
                        env["path"] = Path(env["path"][1:])  # Remove the first point

                        # If path is empty, we've reached the destination
                        if len(env["path"]) == 0:
                            print(f"Reached final destination: {env['destination']}")
                            env["is_navigating"] = False

                time.sleep(0.1)

            print("Automatic navigation completed")
            return True
        except KeyboardInterrupt:
            print("\nAutomatic navigation interrupted")
            return False

    drawer = Drawer(interactive=True, dark_mode=True)
    drawer.on_click = lambda x: navigate_to(x) and draw()

    def draw():
        costmap = env["costmap"]
        position = env["position"]
        destination = env["destination"]
        target = env["target"]
        path = env["path"]
        drawer.clear(title="A* Path Planning")
        drawer.draw(
            (costmap, {"transparent_unknown": False}),
            position,
            (target, {"color": "#ff0000", "markersize": 5, "marker": "x"}),
            (destination, {"color": "#00ff00"}),
            (path, {"color": "#ff0000"}),
        )

    while True:
        draw()
        plt.pause(0.1)

    try:
        # Import IPython for interactive REPL
        from IPython import embed

        print("\n=== Manual Robot Control (IPython) ===")
        print("Available commands:")
        print("- move(x, y, yaw=0.0, duration=2.0): Move robot with velocity")
        print("- pos(): Show current position")
        print("- nav(x, y): Navigate to position using A*")
        print("- auto(): Run automatic navigation mode")
        print("\nPress Ctrl+D to exit at any time")

        # Create the namespace for IPython
        local_vars = {
            #            "robot": robot,
            "env": env,
            "Vector": Vector,
            "move": move,
            "pos": pos,
            "nav": nav,
            "auto": auto,
            "walk": walk,
            "draw": draw,
            "navigate_to": navigate_to,
        }
        embed(local_ns=local_vars)

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

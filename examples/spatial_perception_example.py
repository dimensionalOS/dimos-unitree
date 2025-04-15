
import os
import sys
import time
import rclpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from dimos.perception.spatial_perception import SpatialPerception

def main():
    rclpy.init()
    
    ros_control = UnitreeROSControl(
        node_name="spatial_perception_example"
    )
    
    robot = UnitreeGo2(ros_control=ros_control)
    
    video_stream = robot.get_ros_video_stream()
    
    position_stream = ros_control.get_position_stream()
    
    spatial_perception = SpatialPerception(
        collection_name="robot_spatial_memory",
        min_distance_threshold=0.5,  # Store frames every 0.5 meters
        min_time_threshold=2.0  # Store frames at least every 2 seconds
    )
    
    result_stream = spatial_perception.process_video_stream(
        video_stream=video_stream,
        position_stream=position_stream
    )
    
    def on_result(result):
        print(f"Stored frame at position: ({result['position'][0]:.2f}, {result['position'][1]:.2f})")
    
    result_stream.subscribe(on_result)
    
    print("Running spatial perception for 5 minutes...")
    try:
        start_time = time.time()
        while time.time() - start_time < 300:  # 5 minutes
            rclpy.spin_once(ros_control._node, timeout_sec=0.1)
    
    except KeyboardInterrupt:
        print("Example interrupted by user")
    
    spatial_perception.cleanup()
    robot.cleanup()
    rclpy.shutdown()
    
    print("Example completed successfully")

if __name__ == "__main__":
    main()

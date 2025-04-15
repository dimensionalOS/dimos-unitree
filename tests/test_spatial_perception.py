
import os
import sys
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl
from dimos.perception.spatial_perception import SpatialPerception

def main():
    rclpy.init()
    
    ros_control = UnitreeROSControl(
        node_name="spatial_perception_test",
        mock_connection=True  # Set to False when testing on a real robot
    )
    
    robot = UnitreeGo2(ros_control=ros_control)
    
    video_stream = robot.get_ros_video_stream()
    
    position_stream = ros_control.get_position_stream()
    
    spatial_perception = SpatialPerception(
        collection_name="test_spatial_memory",
        min_distance_threshold=0.2,  # Store frames every 0.2 meters
        min_time_threshold=1.0  # Store frames at least every 1 second
    )
    
    result_stream = spatial_perception.process_video_stream(
        video_stream=video_stream,
        position_stream=position_stream
    )
    
    def on_result(result):
        print(f"Stored frame at position: ({result['position'][0]:.2f}, {result['position'][1]:.2f})")
        
        cv2.imshow("Stored Frame", result["frame"])
        cv2.waitKey(1)
    
    result_stream.subscribe(on_result)
    
    print("Running spatial perception for 60 seconds...")
    try:
        start_time = time.time()
        while time.time() - start_time < 60:
            rclpy.spin_once(ros_control._node, timeout_sec=0.1)
            
            t = time.time() - start_time
            x = 2.0 * np.cos(t * 0.2)
            y = 2.0 * np.sin(t * 0.2)
            
            # This avoids accessing internal attributes of ROSControl
            from reactivex import Subject
            
            if not hasattr(spatial_perception, '_test_position_subject'):
                spatial_perception._test_position_subject = Subject()
                spatial_perception._test_position_subject.subscribe(
                    lambda pos: setattr(spatial_perception, 'current_position', pos)
                )
            
            spatial_perception._test_position_subject.on_next((x, y))
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    cv2.destroyAllWindows()
    
    print("\nQuerying by location (0, 0)...")
    location_results = spatial_perception.query_by_location(0, 0, radius=1.0, limit=3)
    
    for i, result in enumerate(location_results):
        print(f"Result {i+1}: Position ({result['metadata']['x']:.2f}, {result['metadata']['y']:.2f}), "
              f"Distance: {result.get('distance', 'N/A')}")
        
        cv2.imshow(f"Location Result {i+1}", result["image"])
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    if location_results:
        print("\nQuerying by image similarity...")
        image_results = spatial_perception.query_by_image(location_results[0]["image"], limit=3)
        
        for i, result in enumerate(image_results):
            print(f"Result {i+1}: Position ({result['metadata']['x']:.2f}, {result['metadata']['y']:.2f}), "
                  f"Distance: {result.get('distance', 'N/A')}")
            
            cv2.imshow(f"Image Result {i+1}", result["image"])
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    
    locations = spatial_perception.vector_db.get_all_locations()
    
    if locations:
        x_coords, y_coords = zip(*locations)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7)
        plt.title("Spatial Memory Map")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.grid(True)
        plt.axis('equal')
        
        plt.gca().add_patch(Circle((0, 0), 1.0, fill=False, color='red', linestyle='--'))
        
        plt.savefig("spatial_memory_map.png")
        plt.show()
    
    spatial_perception.cleanup()
    robot.cleanup()
    rclpy.shutdown()
    
    print("Test completed successfully")

if __name__ == "__main__":
    main()

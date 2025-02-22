#!/usr/bin/env python3

import cv2
from dimos.robot.unitree.unitree_go2 import UnitreeGo2, WebRTCConnectionMethod
import os
import time
from dimos.robot.unitree.unitree_ros_control import UnitreeROSControl

def get_env_var(var_name, default=None, required=False):
    """Get environment variable with validation."""
    value = os.getenv(var_name, default)
    if required and not value:
        raise ValueError(f"{var_name} environment variable is required")
    return value

if __name__ == "__main__":
    # Get configuration from environment variables
    robot_ip = get_env_var("ROBOT_IP", "192.168.9.140")
    connection_method = get_env_var("CONNECTION_METHOD", "LocalSTA")
    serial_number = get_env_var("SERIAL_NUMBER", None)
    output_dir = "/dimos/assets/output"
    api_call_interval = int(get_env_var("API_CALL_INTERVAL", "5"))

    # Convert connection method string to enum
    connection_method = getattr(WebRTCConnectionMethod, connection_method)

    print("Initializing UnitreeGo2...")
    print(f"Configuration:")
    print(f"  IP: {robot_ip}")
    print(f"  Connection Method: {connection_method}")
    print(f"  Serial Number: {serial_number if serial_number else 'Not provided'}")
    print(f"  Output Directory: {output_dir}")
    print(f"  API Call Interval: {api_call_interval} seconds")

    robot = UnitreeGo2(
        ip=robot_ip,
        connection_method=connection_method,
        serial_number=serial_number,
        output_dir=output_dir,
        api_call_interval=api_call_interval,
        ros_control=UnitreeROSControl()
    )
    
    try:
        def start_perception():
            # Start perception
            print("\nStarting perception system...")
            robot.start_perception()        
        
        def example_movement_sequence():
            # Example movement sequence
            print("\nExecuting movement sequence...")
            print("Moving forward...")
            robot.move(0.1, 0.0, 0.0, duration=5.0)  # Move forward for 5 seconds
            time.sleep(0.5)
            
            print("Moving left...")
            robot.move(0.0, 0.3, 0.0, duration=1.0)  # Move left for 1 second
            time.sleep(0.5)
            
            print("Rotating...")
            robot.move(0.0, 0.0, 0.5, duration=5.0)  # Rotate for 5 second
            time.sleep(0.5)
            
            print("\nMonitoring agent outputs (Press Ctrl+C to stop)...")
            # Monitor agent outputs every 5 seconds
        
        # example_movement_sequence()
        start_perception()
        
        while True:
            time.sleep(5)
            # robot.read_agent_outputs()
            
    except KeyboardInterrupt:
        print("\nStopping perception...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        print("Cleaning up resources...")
        del robot
        print("Cleanup complete.") 
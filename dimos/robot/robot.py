from abc import ABC, abstractmethod
from dimos.hardware.interface import HardwareInterface
from dimos.agents.agent_config import AgentConfig
from dimos.robot.ros_control import ROSControl
from dimos.stream.frame_processor import FrameProcessor
from dimos.stream.video_operators import VideoOperators as vops
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
import os
import time
import logging

import multiprocessing

'''
Base class for all dimos robots, both physical and simulated.
'''
class Robot(ABC):
    def __init__(self,
                 agent_config: AgentConfig = None,
                 hardware_interface: HardwareInterface = None,
                 ros_control: ROSControl = None,
                 output_dir: str = os.path.join(os.getcwd(), "output")):
        
        self.agent_config = agent_config
        self.hardware_interface = hardware_interface
        self.ros_control = ros_control
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def start_ros_perception(self, fps: int = 30, save_frames: bool = True):
        """Start ROS-based perception system with rate limiting and frame processing."""
        if not self.ros_control or not self.ros_control.video_provider:
            raise RuntimeError("No ROS video provider available")
            
        print(f"Starting ROS video stream at {fps} FPS...")
        
        # Get base stream from video provider
        video_stream = self.ros_control.video_provider.capture_video_as_observable(fps=fps)
        
        # Add minimal processing pipeline
        processed_stream = video_stream.pipe(
            ops.do_action(lambda x: print(f"ROBOT: Received frame of type {type(x)}")),
            ops.catch(lambda e: print(f"ROBOT: Error in stream: {e}")),
            ops.share()
        )
        
        # Add debug subscription
        processed_stream.subscribe(
            on_next=lambda x: print("ROBOT: Frame ready for final subscriber"),
            on_error=lambda e: print(f"ROBOT: Pipeline error: {e}")
        )
        
        return processed_stream

    def move(self, x: float, y: float, yaw: float, duration: float = 0.0) -> bool:
        """Move the robot using velocity commands.
        
        Args:
            x: Forward/backward velocity (m/s)
            y: Left/right velocity (m/s)
            yaw: Rotational velocity (rad/s)
            duration: How long to move (seconds). If 0, command is continuous
            
        Returns:
            bool: True if command was sent successfully
        """
        if self.ros_control is None:
            raise RuntimeError("No ROS control interface available for movement")
        return self.ros_control.move(x, y, yaw, duration)

    @abstractmethod
    def do(self, *args, **kwargs):
     """Executes motion."""
    pass
    def update_hardware_interface(self, new_hardware_interface: HardwareInterface):
        """Update the hardware interface with a new configuration."""
        self.hardware_interface = new_hardware_interface

    def get_hardware_configuration(self):
        """Retrieve the current hardware configuration."""
        return self.hardware_interface.get_configuration()

    def set_hardware_configuration(self, configuration):
        """Set a new hardware configuration."""
        self.hardware_interface.set_configuration(configuration)

    def cleanup(self):
        """Cleanup resources."""
        if self.ros_control:
            self.ros_control.cleanup()

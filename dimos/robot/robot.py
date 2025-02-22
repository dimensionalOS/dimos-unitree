from abc import ABC, abstractmethod
from dimos.hardware.interface import HardwareInterface
from dimos.agents.agent_config import AgentConfig
from dimos.robot.ros_control import ROSControl
from dimos.stream.frame_processor import FrameProcessor
from dimos.stream.video_operators import VideoOperators as vops
from reactivex import operators as ops
import os
import time
import logging

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
        """Start ROS-based perception system with rate limiting and frame processing.
        
        Args:
            fps: Frames per second to process
            save_frames: Whether to save frames to disk
        """
        if not self.ros_control or not self.ros_control.data_provider:
            raise RuntimeError("No ROS data provider available")
            
        print(f"Starting ROS data stream at {fps} FPS...")
        
        # Create data stream observable with desired FPS
        data_stream_obs = self.ros_control.data_provider.capture_data_as_observable(fps=fps)
        
        # Create frame counter
        def create_frame_counter():
            count = 0
            def increment():
                nonlocal count
                count += 1
                return count
            return increment
        
        frame_counter = create_frame_counter()
        
        # Initialize frame processor if saving frames
        frame_processor = None
        if save_frames:
            frame_processor = FrameProcessor(
                delete_on_init=True,
                output_dir=os.path.join(self.output_dir, "frames")
            )
        
        # Process the data stream
        processed_stream = data_stream_obs.pipe(
            # Add frame counting
            ops.do_action(lambda _: print(f"Frame {frame_counter()} received")),
            # Save frames if requested
            *([vops.with_jpeg_export(frame_processor, suffix="ros_frame_", save_limit=100)] if save_frames else []),
            # Add error handling
            ops.catch(lambda e, _: print(f"Error in stream processing: {e}")),
            # Share the stream among multiple subscribers
            ops.share()
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

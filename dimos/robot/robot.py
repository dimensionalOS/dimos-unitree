# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base module for all DIMOS robots.

This module provides the foundation for all DIMOS robots, including both physical 
and simulated implementations, with common functionality for movement, control, 
and video streaming.
"""

from abc import ABC
import os
import logging
from typing import TYPE_CHECKING, Optional, Dict, Tuple, Any

import chromadb
from dimos.hardware.interface import HardwareInterface
from dimos.perception.spatial_perception import SpatialMemory
from dimos.agents.memory.visual_memory import VisualMemory
from dimos.types.robot_location import RobotLocation
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.robot.ros_control import ROSControl
else:
    ROSControl = 'ROSControl'

from dimos.skills.skills import SkillLibrary
from dimos.stream.frame_processor import FrameProcessor
from dimos.stream.video_operators import VideoOperators as vops
from reactivex import Observable, operators as ops
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import ThreadPoolScheduler

from dimos.utils.threadpool import get_scheduler

logger = setup_logger("dimos.robot.robot")

class Robot(ABC):
    """Base class for all DIMOS robots.
    
    This abstract base class defines the common interface and functionality for all
    DIMOS robots, whether physical or simulated. It provides methods for movement,
    rotation, video streaming, and hardware configuration management.
    
    Attributes:
        agent_config: Configuration for the robot's agent.
        hardware_interface: Interface to the robot's hardware components.
        ros_control: ROS-based control system for the robot.
        output_dir: Directory for storing output files.
        disposables: Collection of disposable resources for cleanup.
        pool_scheduler: Thread pool scheduler for managing concurrent operations.
    """

    def __init__(self,
                 hardware_interface: HardwareInterface = None,
                 ros_control: ROSControl = None,
                 output_dir: str = os.path.join(os.getcwd(), "assets", "output"),
                 pool_scheduler: ThreadPoolScheduler = None,
                 skill_library: SkillLibrary = None,
                 spatial_memory_dir: str = None,
                 spatial_memory_collection: str = "spatial_memory"):
        """Initialize a Robot instance.
        
        Args:
            hardware_interface: Interface to the robot's hardware. Defaults to None.
            ros_control: ROS-based control system. Defaults to None.
            output_dir: Directory for storing output files. Defaults to "./assets/output".
            pool_scheduler: Thread pool scheduler. If None, one will be created.
            skill_library: Skill library instance. If None, one will be created.
            spatial_memory_dir: Directory for storing spatial memory data. If None, uses output_dir/spatial_memory.
            spatial_memory_collection: Name of the collection in the ChromaDB database.
        """
        self.hardware_interface = hardware_interface
        self.ros_control = ros_control
        self.output_dir = output_dir
        self.disposables = CompositeDisposable()
        self.pool_scheduler = pool_scheduler if pool_scheduler else get_scheduler()
        self.skill_library = skill_library if skill_library else SkillLibrary()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize spatial memory properties
        self.spatial_memory_dir = spatial_memory_dir or os.path.join(self.output_dir, "spatial_memory")
        self.spatial_memory_collection = spatial_memory_collection
        self.db_path = os.path.join(self.spatial_memory_dir, "chromadb_data")
        self.visual_memory_path = os.path.join(self.spatial_memory_dir, "visual_memory.pkl")
        
        # Create spatial memory directory
        os.makedirs(self.spatial_memory_dir, exist_ok=True)
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize SpatialMemory immediately
        self._spatial_memory = None
        self._spatial_memory_subscription = None
        self._init_spatial_memory()

    def get_ros_video_stream(self, fps: int = 30) -> Observable:
        """Get the ROS video stream with rate limiting and frame processing.
        
        Args:
            fps: Frames per second for the video stream. Defaults to 30.
            
        Returns:
            Observable: An observable stream of video frames.
            
        Raises:
            RuntimeError: If no ROS video provider is available.
        """
        if not self.ros_control or not self.ros_control.video_provider:
            raise RuntimeError("No ROS video provider available")

        print(f"Starting ROS video stream at {fps} FPS...")

        # Get base stream from video provider
        video_stream = self.ros_control.video_provider.capture_video_as_observable(
            fps=fps)

        # Add minimal processing pipeline with proper thread handling
        processed_stream = video_stream.pipe(
            ops.subscribe_on(self.pool_scheduler),
            ops.observe_on(self.pool_scheduler),  # Ensure thread safety
            ops.share()  # Share the stream
        )

        return processed_stream

    def move(self, distance: float, speed: float = 0.5) -> bool:
        """Move the robot using velocity commands.
        
        DEPRECATED: Use move_vel instead for direct velocity control.
        
        Args:
            distance: Distance to move forward in meters (must be positive).
            speed: Speed to move at in m/s. Defaults to 0.5.
            
        Returns:
            bool: True if movement succeeded.
            
        Raises:
            RuntimeError: If no ROS control interface is available.
        """
        pass

    def reverse(self, distance: float, speed: float = 0.5) -> bool:
        """Move the robot backward by a specified distance.
        
        DEPRECATED: Use move_vel with negative x value instead for direct velocity control.
        
        Args:
            distance: Distance to move backward in meters (must be positive).
            speed: Speed to move at in m/s. Defaults to 0.5.
            
        Returns:
            bool: True if movement succeeded.
            
        Raises:
            RuntimeError: If no ROS control interface is available.
        """
        pass

    def spin(self, degrees: float, speed: float = 45.0) -> bool:
        """Rotate the robot by a specified angle.

        Args:
            degrees: Angle to rotate in degrees (positive for counter-clockwise, 
                negative for clockwise).
            speed: Angular speed in degrees/second. Defaults to 45.0.
            
        Returns:
            bool: True if rotation succeeded.
            
        Raises:
            RuntimeError: If no ROS control interface is available.
        """
        if self.ros_control is None:
            raise RuntimeError(
                "No ROS control interface available for rotation")
        return self.ros_control.spin(degrees, speed)

    def webrtc_req(self, api_id: int, topic: str = None, parameter: str = '', 
                  priority: int = 0, request_id: str = None, data=None, timeout: float = 1000.0) -> bool:
        """Send a WebRTC request command to the robot.
        
        Args:
            api_id: The API ID for the command.
            topic: The API topic to publish to. Defaults to ROSControl.webrtc_api_topic.
            parameter: Optional parameter string. Defaults to ''.
            priority: Priority level as defined by PriorityQueue(). Defaults to 0 (no priority).
            data: Optional data dictionary.
            timeout: Maximum time to wait for the command to complete.

        Returns:
            bool: True if command was sent successfully.
            
        Raises:
            RuntimeError: If no ROS control interface is available.

        """
        if self.ros_control is None:
            raise RuntimeError("No ROS control interface available for WebRTC commands")
        return self.ros_control.queue_webrtc_req(
            api_id=api_id, 
            topic=topic,
            parameter=parameter, 
            priority=priority,
            request_id=request_id,
            data=data,
            timeout=timeout
        )

    def move_vel(self, x: float, y: float, yaw: float, duration: float = 0.0) -> bool:
        """Move the robot using direct movement commands.
        
        Args:
            x: Forward/backward velocity (m/s)
            y: Left/right velocity (m/s)
            yaw: Rotational velocity (rad/s)
            duration: How long to move (seconds). If 0, command is continuous
            
        Returns:
            bool: True if command was sent successfully
            
        Raises:
            RuntimeError: If no ROS control interface is available.
        """
        if self.ros_control is None:
            raise RuntimeError("No ROS control interface available for movement")
        return self.ros_control.move_vel(x, y, yaw, duration)
    
    def pose_command(self, roll: float, pitch: float, yaw: float) -> bool:
        """Send a pose command to the robot.
        
        Args:
            roll: Roll angle in radians.
            pitch: Pitch angle in radians.
            yaw: Yaw angle in radians.
            
        Returns:
            bool: True if command was sent successfully.
            
        Raises:
            RuntimeError: If no ROS control interface is available.
        """
        if self.ros_control is None:
            raise RuntimeError("No ROS control interface available for pose commands")
        return self.ros_control.pose_command(roll, pitch, yaw)

    def update_hardware_interface(self,
                                  new_hardware_interface: HardwareInterface):
        """Update the hardware interface with a new configuration.
        
        Args:
            new_hardware_interface: New hardware interface to use for the robot.
        """
        self.hardware_interface = new_hardware_interface

    def get_hardware_configuration(self):
        """Retrieve the current hardware configuration.
        
        Returns:
            The current hardware configuration from the hardware interface.
            
        Raises:
            AttributeError: If hardware_interface is None.
        """
        return self.hardware_interface.get_configuration()

    def set_hardware_configuration(self, configuration):
        """Set a new hardware configuration.
        
        Args:
            configuration: The new hardware configuration to set.
            
        Raises:
            AttributeError: If hardware_interface is None.
        """
        self.hardware_interface.set_configuration(configuration)

    def _init_spatial_memory(self, new_map: bool = False, min_distance_threshold: float = 0.01, min_time_threshold: float = 1.0) -> None:
        """Initialize the spatial memory instance and start continuous processing.
        
        Args:
            new_map: If True, creates a new spatial memory from scratch
            min_distance_threshold: Minimum distance in meters to record a new frame
            min_time_threshold: Minimum time in seconds to record a new frame
        """
        # Clean up any existing subscription
        if self._spatial_memory_subscription is not None and not self._spatial_memory_subscription.is_disposed:
            logger.info("Disposing existing spatial memory subscription")
            self._spatial_memory_subscription.dispose()
            self._spatial_memory_subscription = None
            
        # Set up persistent ChromaDB
        if new_map and os.path.exists(self.db_path):
            try:
                logger.info(f"Creating new ChromaDB database (new_map=True)")
                # Try to delete any existing database files
                import shutil
                for item in os.listdir(self.db_path):
                    item_path = os.path.join(self.db_path, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                logger.info(f"Removed existing ChromaDB files from {self.db_path}")
            except Exception as e:
                logger.error(f"Error clearing ChromaDB directory: {e}")
        
        db_client = chromadb.PersistentClient(path=self.db_path)
        
        # Setup visual memory
        if new_map or not os.path.exists(self.visual_memory_path):
            logger.info("Creating new visual memory")
            visual_memory = VisualMemory(output_dir=self.spatial_memory_dir)
        else:
            try:
                logger.info(f"Loading existing visual memory from {self.visual_memory_path}...")
                visual_memory = VisualMemory.load(self.visual_memory_path, output_dir=self.spatial_memory_dir)
                logger.info(f"Loaded {visual_memory.count()} images from previous runs")
            except Exception as e:
                logger.error(f"Error loading visual memory: {e}")
                visual_memory = VisualMemory(output_dir=self.spatial_memory_dir)
        
        # Create spatial memory instance
        self._spatial_memory = SpatialMemory(
            collection_name=self.spatial_memory_collection,
            min_distance_threshold=min_distance_threshold,
            min_time_threshold=min_time_threshold,
            chroma_client=db_client,
            visual_memory=visual_memory
        )
        
        # Store the visual memory instance for saving later
        self._visual_memory = visual_memory
        
        # Start continuous background processing of video frames if ROS control is available
        if self.ros_control is not None and self.ros_control.video_provider is not None:
            logger.info("Starting continuous spatial memory processing")
            try:
                # Get video stream
                video_stream = self.get_ros_video_stream(fps=10)  # Lower FPS to reduce processing load
                
                # Map each video frame to include both position and rotation from transform
                combined_stream = video_stream.pipe(
                    ops.map(lambda video_frame: {
                        "frame": video_frame,
                        **self._extract_transform_data(*self.ros_control.transform_euler("base_link"))
                    })
                )
                
                # Process with spatial memory
                result_stream = self._spatial_memory.process_stream(combined_stream)
                
                # Subscribe to the result stream
                self._spatial_memory_subscription = result_stream.subscribe(
                    on_next=self._on_spatial_memory_update,
                    on_error=lambda e: logger.error(f"Error in spatial memory stream: {e}"),
                    on_completed=lambda: logger.info("Spatial memory stream completed")
                )
                
                logger.info("Continuous spatial memory processing started successfully")
            except Exception as e:
                logger.error(f"Failed to start continuous spatial memory processing: {e}")
        else:
            logger.warning("ROS control or video provider not available, spatial memory will not process video frames")
    
    def _extract_transform_data(self, position, rotation):
        """Extract position and rotation from a transform message."""
        if position is None or rotation is None:
            return {
                "position": None,
                "rotation": None
            }
        return {
            "position": position,
            "rotation": rotation
        }
    
    def _on_spatial_memory_update(self, result):
        """Handle updates from the spatial memory processing stream."""
        # Log successful frame storage (if stored)
        if result.get('stored', True) != False:
            position = result['position']
            logger.debug(f"Spatial memory updated with frame at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            
            # Periodically save visual memory to disk (e.g., every 100 frames)
            if hasattr(self, '_visual_memory') and self._visual_memory is not None:
                if self._spatial_memory.stored_frame_count % 100 == 0:
                    self.save_spatial_memory()
    
    def get_spatial_memory(self, new_map: bool = False, min_distance_threshold: float = 0.01, min_time_threshold: float = 1.0) -> SpatialMemory:
        """Get the spatial memory instance, initializing or reinitializing if needed.
        
        Args:
            new_map: If True, reinitializes spatial memory from scratch
            min_distance_threshold: Minimum distance in meters to record a new frame
            min_time_threshold: Minimum time in seconds to record a new frame
            
        Returns:
            The spatial memory instance
        """
        if self._spatial_memory is None or new_map:
            self._init_spatial_memory(new_map, min_distance_threshold, min_time_threshold)
        
        return self._spatial_memory
    
    def save_spatial_memory(self) -> bool:
        """Save the visual memory component of spatial memory to disk.
        
        Returns:
            True if memory was saved successfully, False otherwise
        """
        if hasattr(self, '_visual_memory') and self._visual_memory is not None:
            visual_memory_filename = os.path.basename(self.visual_memory_path)
            saved_path = self._visual_memory.save(visual_memory_filename)
            logger.info(f"Saved {self._visual_memory.count()} images to disk at {saved_path}")
            return True
        return False
        
    def cleanup(self):
        """Clean up resources used by the robot.
        
        This method should be called when the robot is no longer needed to
        ensure proper release of resources such as ROS connections and
        subscriptions.
        """
        # Dispose of spatial memory subscription if active
        if hasattr(self, '_spatial_memory_subscription') and self._spatial_memory_subscription is not None:
            try:
                logger.info("Disposing spatial memory subscription")
                self._spatial_memory_subscription.dispose()
                self._spatial_memory_subscription = None
            except Exception as e:
                logger.error(f"Error disposing spatial memory subscription: {e}")
        
        # Save spatial memory if it exists
        self.save_spatial_memory()
        
        # Clean up spatial memory if it exists
        if self._spatial_memory:
            self._spatial_memory.cleanup()
            self._spatial_memory = None
        
        if self.ros_control:
            self.ros_control.cleanup()
        self.disposables.dispose()

class MockRobot(Robot):
    def __init__(self):
        super().__init__()
        self.ros_control = None
        self.hardware_interface = None
        self.skill_library = SkillLibrary()

    def my_print(self):
        print("Hello, world!")
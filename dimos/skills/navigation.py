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

"""
Semantic map skills for building and navigating spatial memory maps.

This module provides two skills:
1. BuildSemanticMap - Builds a semantic map by recording video frames at different locations
2. Navigate - Queries an existing semantic map using natural language
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import json
from typing import Optional, Dict, Tuple, Any
from dimos.utils.threadpool import get_scheduler

import chromadb
from reactivex import operators as ops
from reactivex.subject import Subject
from pydantic import Field

from dimos.skills.skills import AbstractRobotSkill
from dimos.perception.spatial_perception import SpatialMemory
from dimos.agents.memory.visual_memory import VisualMemory
from dimos.types.robot_location import RobotLocation
from dimos.utils.threadpool import get_scheduler
from dimos.utils.logging_config import setup_logger
from dimos.models.qwen.video_query import get_bbox_from_qwen_frame
from dimos.utils.generic_subscriber import GenericSubscriber
from dimos.utils.ros_utils import distance_angle_to_goal_xy
from dimos.robot.local_planner.local_planner import navigate_to_goal_local

logger = setup_logger("dimos.skills.semantic_map_skills")

def get_dimos_base_path():
    """
    Get the DiMOS base path from DIMOS_PATH environment variable or default to user's home directory.
    
    Returns:
        Base path to use for DiMOS assets
    """
    dimos_path = os.environ.get('DIMOS_PATH')
    if dimos_path:
        return dimos_path
    # Get the current user's username
    user = os.environ.get('USER', os.path.basename(os.path.expanduser('~')))
    return f"/home/{user}/dimos"

class BuildSemanticMap(AbstractRobotSkill):
    """
    A skill that builds a semantic map of the environment by recording video frames
    at different locations as the robot moves.
    
    This skill records video frames and their associated positions, storing them in
    a vector database for later querying. It runs until terminated (Ctrl+C).
    """
    
    min_distance_threshold: float = Field(0.01, 
                                        description="Min distance in meters to record a new frame")
    min_time_threshold: float = Field(1.0, 
                                    description="Min time in seconds to record a new frame")
    new_map: bool = Field(False,
                        description="If True, creates new spatial and visual memory from scratch instead of using existing.")

    def __init__(self, robot=None, **data):
        """
        Initialize the BuildSemanticMap skill.
        
        Args:
            robot: The robot instance
            **data: Additional data for configuration
        """
        super().__init__(robot=robot, **data)
        self._stop_event = threading.Event()
        self._subscription = None
        self._scheduler = get_scheduler()
        self._stored_count = 0
        
    def __call__(self):
        """
        Start building a semantic map.
        
        Returns:
            A message indicating whether the map building started successfully
        """
        super().__call__()
        
        if self._robot is None:
            error_msg = "No robot instance provided to BuildSemanticMap skill"
            logger.error(error_msg)
            return error_msg
        
        self.stop()  # Stop any existing execution
        self._stop_event.clear()
        self._stored_count = 0
        
        # Get the ros_control instance from the robot
        ros_control = self._robot.ros_control
        
        # Get or initialize spatial memory from the robot
        logger.info("Getting SpatialMemory from robot...")
        spatial_memory = self._robot.get_spatial_memory(
            new_map=self.new_map,
            min_distance_threshold=self.min_distance_threshold,
            min_time_threshold=self.min_time_threshold
        )
        
        logger.info("Setting up video stream...")
        video_stream = self._robot.get_ros_video_stream()
        
        # Setup video stream processing with transform acquisition
        logger.info("Setting up video stream with position mapping...")
        
        # Map each video frame to include both position and rotation from transform
        combined_stream = video_stream.pipe(
            ops.map(lambda video_frame: {
                "frame": video_frame,
                **self._extract_transform_data(*ros_control.transform_euler("base_link"))
            })
        )
        
        # Process with spatial memory
        result_stream = spatial_memory.process_stream(combined_stream)
        
        # Subscribe to the result stream
        logger.info("Subscribing to spatial perception results...")
        self._subscription = result_stream.subscribe(
            on_next=self._on_stored_frame,
            on_error=lambda e: logger.error(f"Error in spatial memory stream: {e}"),
            on_completed=lambda: logger.info("Spatial memory stream completed")
        )
        
        # Store the spatial memory instance for later cleanup
        self._spatial_memory = spatial_memory
        self._visual_memory = visual_memory
        
        skill_library = self._robot.get_skills()
        self.register_as_running("BuildSemanticMap", skill_library, self._subscription)
        
        logger.info(f"BuildSemanticMap started with min_distance={self.min_distance_threshold}m, "
                 f"min_time={self.min_time_threshold}s")
        return (f"BuildSemanticMap started. Recording frames with min_distance={self.min_distance_threshold}m, "
                f"min_time={self.min_time_threshold}s. Press Ctrl+C to stop.")
    


    def _extract_transform_data(self, position, rotation):
        """
        Extract both position and rotation data from a transform message in a single operation.
        
        Args:
            position: The position message
            rotation: The rotation message
            
        Returns:
            A dictionary containing:
            - 'position': tuple of (x, y, z) coordinates
            - 'rotation': the quaternion object for complete orientation data
        """
        if position is None or rotation is None:
            return {
                "position": None,
                "rotation": None
            }

        return {
            "position": position,
            "rotation": rotation
        }
    
    def _on_stored_frame(self, result):
        """
        Callback for when a frame is stored in the vector database.
        
        Args:
            result: The result of storing the frame
        """
        # Only count actually stored frames (not debug frames)
        if not result.get('stored', True) == False:
            self._stored_count += 1
            pos = result['position']
            logger.info(f"Stored frame #{self._stored_count} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    def stop(self):
        """
        Stop building the semantic map.
        
        Returns:
            A message indicating whether the map building was stopped successfully
        """
        if self._subscription is not None and not self._subscription.is_disposed:
            logger.info("Stopping BuildSemanticMap")
            self._stop_event.set()
            self._subscription.dispose()
            self._subscription = None
            
            # Save spatial memory (visual memory) to disk for later use
            self._robot.save_spatial_memory()
            
            # Clean up spatial memory
            if hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
                self._spatial_memory.cleanup()
                self._spatial_memory = None
                        
            return f"BuildSemanticMap stopped. Stored {self._stored_count} frames."
        return "BuildSemanticMap was not running."


class Navigate(AbstractRobotSkill):
    """
    A skill that queries an existing semantic map using natural language or tries to navigate to an object in view.
    
    This skill first attempts to locate an object in the robot's camera view using vision.
    If the object is found, it navigates to it. If not, it falls back to querying the 
    semantic map for a location matching the description. For example, "Find the kitchen" 
    will first look for a kitchen in view, then check the semantic map coordinates where 
    a kitchen was previously observed.

    CALL THIS SKILL FOR ONE SUBJECT AT A TIME. For example: "Go to the person wearing a blue shirt in the living room",
    you should call this skill twice, once for the person wearing a blue shirt and once for the living room.
    """
    
    query: str = Field("", description="Text query to search for in the semantic map")

    limit: int = Field(1, description="Maximum number of results to return")
    distance: float = Field(1.0, description="Desired distance to maintain from object in meters")
    timeout: float = Field(40.0, description="Maximum time to spend navigating in seconds")
    similarity_threshold: float = Field(0.25, description="Minimum similarity score required for semantic map results to be considered valid")
    
    def __init__(self, robot=None, **data):
        """
        Initialize the Navigate skill.
        
        Args:
            robot: The robot instance
            **data: Additional data for configuration
        """
        super().__init__(robot=robot, **data)
        self._stop_event = threading.Event()
        self._spatial_memory = None
        self._scheduler = get_scheduler()  # Use the shared DiMOS thread pool
        self._navigation_disposable = None  # Disposable returned by scheduler.schedule()
        self._tracking_subscriber = None  # For object tracking
    
    def _navigate_to_object(self):
        """
        Helper method that attempts to navigate to an object visible in the camera view.
        
        Returns:
            dict: Result dictionary with success status and details
        """
        # Stop any existing operation
        self._stop_event.clear()
        
        try:
            logger.warning(f"Attempting to navigate to visible object: {self.query} with desired distance {self.distance}m, timeout {self.timeout} seconds...")
    
            # Try to get a bounding box from Qwen - only try once
            bbox = None
            try:
                # Capture a single frame from the video stream
                frame = self._robot.get_ros_video_stream().pipe(ops.take(1)).run()
                # Use the frame-based function
                bbox, object_size = get_bbox_from_qwen_frame(frame, object_name=self.query)
            except Exception as e:
                logger.error(f"Error querying Qwen: {e}")
                return {"success": False, "error": f"Could not detect {self.query} in view: {e}"}
    
            if bbox is None or self._stop_event.is_set():
                logger.error(f"Failed to get bounding box for {self.query}")
                return {"success": False, "error": f"Could not find {self.query} in view"}
    
            logger.info(f"Found {self.query} at {bbox} with size {object_size}")
    
            # Start the object tracker with the detected bbox
            self._robot.object_tracker.track(bbox, frame=frame)
            
            # Get the first tracking data with valid distance and angle
            start_time = time.time()
            target_acquired = False
            goal_x_robot = 0
            goal_y_robot = 0
            goal_angle = 0
            
            while time.time() - start_time < 10.0 and not self._stop_event.is_set() and not target_acquired:
                # Get the latest tracking data
                tracking_data = self._robot.object_tracking_stream.pipe(ops.take(1)).run()
                
                if tracking_data and tracking_data.get("targets") and tracking_data["targets"]:
                    target = tracking_data["targets"][0]
                    
                    if "distance" in target and "angle" in target:
                        # Convert target distance and angle to xy coordinates in robot frame
                        goal_distance = target["distance"] - self.distance  # Subtract desired distance to stop short
                        goal_angle = -target["angle"]
                        logger.info(f"Target distance: {goal_distance}, Target angle: {goal_angle}")
                        
                        goal_x_robot, goal_y_robot = distance_angle_to_goal_xy(goal_distance, goal_angle)
                        target_acquired = True
                        break

                    else:
                        logger.warning(f"No valid target tracking data found. target: {target}")

                else:
                    logger.warning(f"No valid target tracking data found. tracking_data: {tracking_data}")
                
                time.sleep(0.1)
            
            if not target_acquired:
                logger.error("Failed to acquire valid target tracking data")
                return {"success": False, "error": "Failed to track object"}
                
            logger.info(f"Navigating to target at local coordinates: ({goal_x_robot:.2f}, {goal_y_robot:.2f}), angle: {goal_angle:.2f}")
            
            # Use navigate_to_goal_local instead of directly controlling the local planner
            success = navigate_to_goal_local(
                robot=self._robot,
                goal_xy_robot=(goal_x_robot, goal_y_robot),
                goal_theta=goal_angle,
                distance=0.0,  # We already accounted for desired distance
                timeout=self.timeout,
                stop_event=self._stop_event
            )
            
            if success:
                logger.info(f"Successfully navigated to {self.query}")
                return {
                    "success": True,
                    "query": self.query,
                    "message": f"Successfully navigated to {self.query} in view"
                }
            else:
                logger.warning(f"Failed to reach {self.query} within timeout or operation was stopped")
                return {
                    "success": False, 
                    "error": f"Failed to reach {self.query} within timeout"
                }
            
        except Exception as e:
            logger.error(f"Error in navigate to object: {e}")
            return {"success": False, "error": f"Error: {e}"}
        finally:
            # Clean up
            self._robot.ros_control.stop()
            self._robot.object_tracker.cleanup()
            
    def _navigate_using_semantic_map(self):
        """
        Helper method that attempts to navigate using the semantic map query.
        
        Returns:
            dict: Result dictionary with success status and details
        """
        logger.info(f"Querying semantic map for: '{self.query}'")
        
        try:
            self._spatial_memory = self._robot.get_spatial_memory()
            
            # Run the query
            results = self._spatial_memory.query_by_text(self.query, limit=self.limit)
            
            if not results:
                logger.warning(f"No results found for query: '{self.query}'")
                return {
                    "success": False,
                    "query": self.query,
                    "error": "No matching location found in semantic map"
                }
            
            # Get the best match
            best_match = results[0]
            metadata = best_match.get('metadata', {})
            
            if isinstance(metadata, list) and metadata:
                metadata = metadata[0]
            
            # Extract coordinates from metadata
            if isinstance(metadata, dict) and 'pos_x' in metadata and 'pos_y' in metadata and 'rot_z' in metadata:
                pos_x = metadata.get('pos_x', 0)
                pos_y = metadata.get('pos_y', 0)
                theta = metadata.get('rot_z', 0)
                
                # Calculate similarity score (distance is inverse of similarity)
                similarity = 1.0 - (best_match.get('distance', 0) if best_match.get('distance') is not None else 0)
                
                logger.info(f"Found match for '{self.query}' at ({pos_x:.2f}, {pos_y:.2f}, rotation {theta:.2f}) with similarity: {similarity:.4f}")
                
                # Check if similarity is below the threshold
                if similarity < self.similarity_threshold:
                    logger.warning(f"Match found but similarity score ({similarity:.4f}) is below threshold ({self.similarity_threshold})")
                    return {
                        "success": False,
                        "query": self.query,
                        "position": (pos_x, pos_y),
                        "rotation": theta,
                        "similarity": similarity,
                        "error": f"Match found but similarity score ({similarity:.4f}) is below threshold ({self.similarity_threshold})"
                    }
                    
                # Reset the stop event before starting navigation
                self._stop_event.clear()
                
                # The scheduler approach isn't working, switch to direct threading
                # Define a navigation function that will run on a separate thread
                def run_navigation():
                    skill_library = self._robot.get_skills()
                    self.register_as_running("Navigate", skill_library)
                    
                    try:
                        logger.info(f"Starting navigation to ({pos_x:.2f}, {pos_y:.2f}) with rotation {theta:.2f}")
                        # Pass our stop_event to allow cancellation
                        result = False
                        try:
                            result = self._robot.global_planner.set_goal((pos_x, pos_y), goal_theta = theta, stop_event=self._stop_event)
                        except Exception as e:
                            logger.error(f"Error calling global_planner.set_goal: {e}")
                            
                        if result:
                            logger.info("Navigation completed successfully")
                        else:
                            logger.error("Navigation did not complete successfully")
                        return result
                    except Exception as e:
                        logger.error(f"Unexpected error in navigation thread: {e}")
                        return False
                    finally:
                        self.stop()
                
                # Cancel any existing navigation before starting a new one
                # Signal stop to any running navigation
                self._stop_event.set()
                # Clear stop event for new navigation
                self._stop_event.clear()
                
                # Run the navigation in the main thread
                run_navigation()

                return {
                    "success": True,
                    "query": self.query,
                    "position": (pos_x, pos_y),
                    "rotation": theta,
                    "similarity": similarity,
                    "metadata": metadata
                }
            else:
                logger.warning(f"No valid position data found for query: '{self.query}'")
                return {
                    "success": False,
                    "query": self.query,
                    "error": "No valid position data found in semantic map"
                }
        except Exception as e:
            logger.error(f"Error in semantic map navigation: {e}")
            return {"success": False, "error": f"Semantic map error: {e}"}
                
    def __call__(self):
        """
        First attempts to navigate to an object in view, then falls back to querying the semantic map.
        
        Returns:
            A dictionary with the result of the navigation attempt
        """
        super().__call__()
        
        if not self.query:
            error_msg = "No query provided to Navigate skill"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # First, try to find and navigate to the object in camera view
        logger.info(f"First attempting to find and navigate to visible object: '{self.query}'")
        object_result = self._navigate_to_object()
        
        if object_result and object_result.get("success", True):
            logger.info(f"Successfully navigated to {self.query} in view")
            return object_result
        
        # If object navigation failed, fall back to semantic map
        logger.info(f"Object not found in view. Falling back to semantic map query for: '{self.query}'")
        
        return self._navigate_using_semantic_map()    


    def stop(self):
        """
        Stop the navigation skill and clean up resources.
        
        Returns:
            A message indicating whether the navigation was stopped successfully
        """
        logger.info("Stopping Navigate skill")
        
        # Signal any running processes to stop via the shared event
        self._stop_event.set()
        
        skill_library = self._robot.get_skills()
        self.unregister_as_running("Navigate", skill_library)
        
        # Dispose of any existing navigation task
        if hasattr(self, '_navigation_disposable') and self._navigation_disposable:
            logger.info("Disposing navigation task")
            try:
                self._navigation_disposable.dispose()
            except Exception as e:
                logger.error(f"Error disposing navigation task: {e}")
            self._navigation_disposable = None
        
        # Clean up spatial memory if it exists
        if hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
            logger.info("Cleaning up spatial memory")
            self._spatial_memory.cleanup()
            self._spatial_memory = None
        
        # Stop robot motion
        self._robot.ros_control.stop()
        
        return "Navigate skill stopped successfully."

class GetPose(AbstractRobotSkill):
    """
    A skill that returns the current position and orientation of the robot.

    This skill is useful for getting the current pose of the robot in the map frame. You call this skill
    if you want to remember a location, for example, "remember this is where my favorite chair is" and then
    call this skill to get the position and rotation of approximately where the chair is. You can then use 
    the position to navigate to the chair.
    
    When location_name is provided, this skill will also remember the current location with that name,
    allowing you to navigate back to it later using the Navigate skill.
    """
    
    location_name: str = Field("", description="Optional name to assign to this location (e.g., 'kitchen', 'office')")
    
    def __init__(self, robot=None, **data):
        """
        Initialize the GetPose skill.
        
        Args:
            robot: The robot instance
            **data: Additional data for configuration
        """
        super().__init__(robot=robot, **data)
    
    def __call__(self):
        """
        Get the current pose of the robot.
        
        Returns:
            A dictionary containing the position and rotation of the robot
        """
        super().__call__()
        
        if self._robot is None:
            error_msg = "No robot instance provided to GetPose skill"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Get the current pose using the robot's get_pose method
            position, rotation = self._robot.get_pose()

            # Format the response
            result = {
                "success": True,
                "position": {
                    "x": position[0],
                    "y": position[1],
                    "z": position[2] if len(position) > 2 else 0.0
                },
                "rotation": {
                    "roll": rotation[0],
                    "pitch": rotation[1],
                    "yaw": rotation[2]
                }
            }
            
            # If location_name is provided, remember this location
            if self.location_name:
                # Get the spatial memory instance
                spatial_memory = self._robot.get_spatial_memory()
                
                # Create a RobotLocation object
                location = RobotLocation(
                    name=self.location_name,
                    position=position,
                    rotation=rotation
                )
                
                # Add to spatial memory
                if spatial_memory.add_robot_location(location):
                    result["location_saved"] = True
                    result["location_name"] = self.location_name
                    logger.info(f"Location '{self.location_name}' saved at {position}")
                else:
                    result["location_saved"] = False
                    logger.error(f"Failed to save location '{self.location_name}'")
            
            return result
        except Exception as e:
            error_msg = f"Error getting robot pose: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    

class NavigateToGoal(AbstractRobotSkill):
    """
    A skill that navigates the robot to a specified position and orientation.
    
    This skill uses the global planner to generate a path to the target position
    and then uses navigate_path_local to follow that path, achieving the desired
    orientation at the goal position.
    """
    
    position: Tuple[float, float] = Field((0.0, 0.0), description="Target position (x, y) in map frame")
    rotation: Optional[float] = Field(None, description="Target orientation (yaw) in radians")
    frame: str = Field("map", description="Reference frame for the position and rotation")
    timeout: float = Field(120.0, description="Maximum time (in seconds) allowed for navigation")
    
    def __init__(self, robot=None, **data):
        """
        Initialize the NavigateToGoal skill.
        
        Args:
            robot: The robot instance
            **data: Additional data for configuration
        """
        super().__init__(robot=robot, **data)
        self._stop_event = threading.Event()
    
    def __call__(self):
        """
        Navigate to the specified goal position and orientation.
        
        Returns:
            A dictionary containing the result of the navigation attempt
        """
        super().__call__()
        
        if self._robot is None:
            error_msg = "No robot instance provided to NavigateToGoal skill"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Reset stop event to make sure we don't immediately abort
        self._stop_event.clear()

        skill_library = self._robot.get_skills()
        self.register_as_running("NavigateToGoal", skill_library)
        
        logger.info(f"Starting navigation to position=({self.position[0]:.2f}, {self.position[1]:.2f}) "
                    f"with rotation={self.rotation if self.rotation is not None else 'None'} "
                    f"in frame={self.frame}")
        
        try:
            # Use the global planner to set the goal and generate a path
            result = self._robot.global_planner.set_goal(
                self.position, 
                goal_theta=self.rotation,
                stop_event=self._stop_event
            )
            
            if result:
                logger.info("Navigation completed successfully")
                return {
                    "success": True,
                    "position": self.position,
                    "rotation": self.rotation,
                    "message": "Goal reached successfully"
                }
            else:
                logger.warning("Navigation did not complete successfully")
                return {
                    "success": False,
                    "position": self.position,
                    "rotation": self.rotation,
                    "message": "Goal could not be reached"
                }
            
        except Exception as e:
            error_msg = f"Error during navigation: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "position": self.position,
                "rotation": self.rotation,
                "error": error_msg
            }
        finally:
            self.stop()
            
    
    def stop(self):
        """
        Stop the navigation.
        
        Returns:
            A message indicating that the navigation was stopped
        """
        logger.info("Stopping NavigateToGoal")
        skill_library = self._robot.get_skills()
        self.unregister_as_running("NavigateToGoal", skill_library)
        self._stop_event.set()
        return "Navigation stopped"

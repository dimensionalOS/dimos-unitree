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
import reactivex
from reactivex import operators as ops
from pydantic import Field

from dimos.skills.skills import AbstractRobotSkill
from dimos.perception.spatial_perception import SpatialMemory
from dimos.agents.memory.visual_memory import VisualMemory
from dimos.types.robot_location import RobotLocation
from dimos.utils.threadpool import get_scheduler
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.skills.semantic_map_skills")

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
    A skill that queries an existing semantic map using natural language.
    
    This skill takes a text query and returns the XY coordinates of the best match
    in the semantic map. For example, "Find the kitchen" will return the coordinates
    where the kitchen was observed.
    """
    
    query: str = Field("", description="Text query to search for in the semantic map")
    limit: int = Field(1, description="Maximum number of results to return")
    
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
    
    def __call__(self):
        """
        Query the semantic map with the provided text query.
        
        Returns:
            A dictionary with the best matching position and query details
        """
        super().__call__()
        
        if not self.query:
            error_msg = "No query provided to Navigate skill"
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"Querying semantic map for: '{self.query}'")
        
        # Get SpatialMemory from robot
        spatial_memory = self._robot.get_spatial_memory()
        
        # Run the query
        results = spatial_memory.query_by_text(self.query, limit=self.limit)
        
        if not results:
            logger.warning(f"No results found for query: '{self.query}'")
            return {
                "success": False,
                "query": self.query,
                "error": "No matching location found"
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
            
            # Create and start direct thread instead of using scheduler
            logger.info("Creating direct navigation thread")
            nav_thread = threading.Thread(target=run_navigation, daemon=True)
            logger.info("Starting direct navigation thread")
            nav_thread.start()
            logger.info("Direct navigation thread started successfully")

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
                "error": "No valid position data found"
            }
    


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

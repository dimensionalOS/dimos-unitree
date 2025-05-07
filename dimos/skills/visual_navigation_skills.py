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
Visual navigation skills for robot interaction.

This module provides skills for visual navigation, including following humans
and navigating to specific objects using computer vision.
"""

import time
import logging
import threading
from typing import Optional, Tuple
import numpy as np

from dimos.skills.skills import AbstractRobotSkill
from dimos.utils.logging_config import setup_logger
from dimos.perception.visual_servoing import VisualServoing
from dimos.models.qwen.video_query import get_bbox_from_qwen_frame
from dimos.utils.generic_subscriber import GenericSubscriber
from dimos.utils.ros_utils import distance_angle_to_goal_xy
from dimos.robot.local_planner.local_planner import navigate_to_goal_local
from pydantic import Field
from reactivex import operators as ops

logger = setup_logger("dimos.skills.visual_navigation", level=logging.DEBUG)

class FollowHuman(AbstractRobotSkill):
    """
    A skill that makes the robot follow a human using visual servoing continuously. 
    
    This skill uses the robot's person tracking stream to follow a human
    while maintaining a specified distance. It will keep following the human
    until the timeout is reached or the skill is stopped. Don't use this skill
    if you want to navigate to a specific person, use NavigateTo instead.
    """
    
    distance: float = Field(1.5, description="Desired distance to maintain from the person in meters")
    timeout: float = Field(20.0, description="Maximum time to follow the person in seconds")
    point: Optional[Tuple[int, int]] = Field(None, description="Optional point to start tracking (x,y pixel coordinates)")
    
    def __init__(self, robot=None, **data):
        super().__init__(robot=robot, **data)
        self._stop_event = threading.Event()
        self._visual_servoing = None
    
    def __call__(self):
        """
        Start following a human using visual servoing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        super().__call__()
        
        if not hasattr(self._robot, 'person_tracking_stream') or self._robot.person_tracking_stream is None:
            logger.error("Robot does not have a person tracking stream")
            return False
        
        # Stop any existing operation
        self.stop()
        self._stop_event.clear()
        
        success = False
        
        try:
            # Initialize visual servoing
            self._visual_servoing = VisualServoing(tracking_stream=self._robot.person_tracking_stream)
            
            logger.warning(f"Following human for {self.timeout} seconds...")
            start_time = time.time()
            
            # Start tracking
            track_success = self._visual_servoing.start_tracking(point=self.point, desired_distance=self.distance)
            
            if not track_success:
                logger.error("Failed to start tracking")
                return False
            
            # Main follow loop
            while (self._visual_servoing.running and 
                   time.time() - start_time < self.timeout and 
                   not self._stop_event.is_set()):
                
                output = self._visual_servoing.updateTracking()
                x_vel = output.get("linear_vel")
                z_vel = output.get("angular_vel")
                logger.debug(f"Following human: x_vel: {x_vel}, z_vel: {z_vel}")
                self._robot.ros_control.move_vel_control(x=x_vel, y=0, yaw=z_vel)
                time.sleep(0.05)
            
            # If we completed the full timeout duration, consider it success
            if time.time() - start_time >= self.timeout:
                success = True
                logger.info("Human following completed successfully")
            elif self._stop_event.is_set():
                logger.info("Human following stopped externally")
            else:
                logger.info("Human following stopped due to tracking loss")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in follow human: {e}")
            return False
        finally:
            # Clean up
            if self._visual_servoing:
                self._visual_servoing.stop_tracking()
                self._visual_servoing = None
            self._robot.ros_control.stop()
    
    def stop(self):
        """
        Stop the human following process.
        
        Returns:
            bool: True if stopped, False if it wasn't running
        """
        if self._visual_servoing is not None:
            logger.info("Stopping FollowHuman skill")
            self._stop_event.set()
            
            # Clean up visual servoing if it exists
            self._visual_servoing.stop_tracking()
            self._visual_servoing = None
            
            # Stop the robot
            self._robot.ros_control.stop()
            
            return True
        return False


class NavigateToObject(AbstractRobotSkill):
    """
    A skill that makes the robot navigate to an object/location/person that is in the view of its camera
    given a description of the object/location/person.

    This skill uses the robot's local planner to navigate to the object/location/person. It gets the bounding
    box of the object/location/person from a VLM and then sends the distance and angle to the local planner.
    USE THIS IF YOU'RE NAVIGATING TO SOMETHING SPECIFIC such as "white door", "person wearing hat". Use this when
    semantic map hasn't been built yet. 
    """
    
    object_name: str = Field(..., description="Name of the object to navigate to")
    distance: float = Field(1.0, description="Desired distance to maintain from object in meters")
    timeout: float = Field(40.0, description="Maximum time to spend navigating in seconds")
    max_retries: int = Field(3, description="Maximum number of retries when getting bounding box")
    
    def __init__(self, robot=None, **data):
        super().__init__(robot=robot, **data)
        self._stop_event = threading.Event()
        self._tracking_subscriber = None
    
    def __call__(self):
        """
        Navigate to an object identified by name using vision-based tracking and local planner.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        super().__call__()
        
        # Stop any existing operation
        self.stop()
        self._stop_event.clear()
        
        success = False
        
        try:
            logger.warning(f"Navigating to {self.object_name} with desired distance {self.distance}m, timeout {self.timeout} seconds...")
    
            # Try to get a bounding box from Qwen with retries
            bbox = None
            retry_count = 0
    
            while bbox is None and retry_count < self.max_retries and not self._stop_event.is_set():
                if retry_count > 0:
                    logger.info(f"Retry {retry_count}/{self.max_retries} to get bounding box for {self.object_name}")
                    # Wait a moment before retry to let the camera feed update
                    time.sleep(1.0)
    
                try:
                    # Capture a single frame from the video stream
                    frame = self._robot.get_ros_video_stream().pipe(ops.take(1)).run()
                    # Use the frame-based function
                    bbox, object_size = get_bbox_from_qwen_frame(frame, object_name=self.object_name)
                except Exception as e:
                    logger.error(f"Error querying Qwen: {e}")
    
                retry_count += 1
    
            if bbox is None or self._stop_event.is_set():
                logger.error(f"Failed to get bounding box for {self.object_name} after {retry_count} attempts")
                return False
    
            logger.info(f"Found {self.object_name} at {bbox} with size {object_size}")
    
            # Start the object tracker with the detected bbox
            self._robot.object_tracker.track(bbox, frame=frame)
    
            # Create a GenericSubscriber to get latest tracking data
            self._tracking_subscriber = GenericSubscriber(self._robot.object_tracking_stream)
    
            # Get the first tracking data with valid distance and angle
            start_time = time.time()
            target_acquired = False
            goal_x_robot = 0
            goal_y_robot = 0
            goal_angle = 0
            
            while time.time() - start_time < 5.0 and not self._stop_event.is_set() and not target_acquired:
                tracking_data = self._tracking_subscriber.get_data()
                
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
                
                time.sleep(0.1)
            
            if not target_acquired:
                logger.error("Failed to acquire valid target tracking data")
                return False
                
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
                logger.info(f"Successfully navigated to {self.object_name}")
            else:
                logger.warning(f"Failed to reach {self.object_name} within timeout or operation was stopped")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in navigate to object: {e}")
            return False
        finally:
            # Clean up
            self._robot.ros_control.stop()
            self._robot.object_tracker.cleanup()
            
            # Clean up tracking subscriber
            if self._tracking_subscriber:
                self._tracking_subscriber.dispose()
                self._tracking_subscriber = None
    
    def stop(self):
        """
        Stop the object navigation process.
        
        Returns:
            bool: True if stopped, False if it wasn't running
        """
        if self._tracking_subscriber is not None:
            logger.info("Stopping NavigateToObject skill")
            self._stop_event.set()
            
            # Stop the robot
            self._robot.ros_control.stop()
            self._robot.object_tracker.cleanup()
            
            # Clean up tracking subscriber if it exists
            self._tracking_subscriber.dispose()
            self._tracking_subscriber = None
            
            return True
        return False

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
State skills for retrieving robot state information.

This module provides the GetState skill, which retrieves the current state
of the robot including position, orientation, and other available information
from the robot's control system.
"""

from typing import Optional, Any, Dict, TYPE_CHECKING
from pydantic import Field

from dimos.skills.skills import AbstractRobotSkill
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.robot.robot import Robot
else:
    Robot = 'Robot'

logger = setup_logger("dimos.skills.state")


class GetState(AbstractRobotSkill):  
    """Get the current state of the robot including position, orientation, and other available state information.
    
    This skill retrieves various state information from the robot, including:
    - Current position and orientation (if available)
    - ROS state data (if available)
    - Battery level, motor temperatures, and other diagnostic information (if available)
    
    The returned state can be used for monitoring the robot's status, making decisions
    based on its current state, or providing feedback to users.
    """
    
    include_details: bool = Field(True, description="Whether to include detailed ROS state information in the response")
      
    def __init__(self, robot: Optional[Robot] = None, **data):  
        super().__init__(robot=robot, **data)  
      
    def __call__(self) -> Dict[str, Any]:  
        """  
        Get the current state of the robot.  
          
        Returns:  
            Dict containing the robot's state information including position, orientation,  
            and any other available state data from the robot's control system.  
        """  
        super().__call__()  # Required call to parent method  
          
        robot_state = {}
        robot_state['success'] = True  
          
        # Get robot position and orientation using get_pose if available  
        if hasattr(self._robot, 'get_pose'):  
            try:  
                position, rotation = self._robot.get_pose()  
                robot_state['position'] = {
                    'x': position[0],
                    'y': position[1],
                    'z': position[2] if len(position) > 2 else 0.0
                }
                robot_state['orientation'] = {
                    'roll': rotation[0],
                    'pitch': rotation[1],
                    'yaw': rotation[2]
                }
            except Exception as e:  
                logger.error(f"Error getting pose: {e}")  
                robot_state['pose_error'] = str(e)
          
        # Get additional state information from ros_control if available  
        if self.include_details and hasattr(self._robot, 'ros_control') and hasattr(self._robot.ros_control, 'get_state'):  
            try:  
                ros_state = self._robot.ros_control.get_state()  
                if ros_state:  
                    robot_state['ros_state'] = ros_state  
            except Exception as e:  
                logger.error(f"Error getting ROS state: {e}")  
                robot_state['ros_state_error'] = str(e)
          
        return robot_state



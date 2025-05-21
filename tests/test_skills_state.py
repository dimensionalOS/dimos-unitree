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

"""Tests for the GetState skill."""

import unittest
from unittest import mock
import tests.test_header

from dimos.skills.state import GetState
from dimos.robot.robot import MockRobot


class TestGetStateSkill(unittest.TestCase):
    """Tests for the GetState skill functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.robot = MockRobot()
        
        # mock get_pose method
        self.robot.get_pose = mock.MagicMock(return_value=((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)))
        
        self.mock_ros_control = mock.MagicMock()
        self.mock_ros_control.get_state = mock.MagicMock(return_value={
            "battery": 0.75,
            "motor_temps": [35.2, 36.1, 37.0, 35.8],
            "cpu_usage": 0.25,
            "status": "operational"
        })
        self.robot.ros_control = self.mock_ros_control
        
    def test_get_state_basic(self):
        """Test the basic functionality of the GetState skill."""
        # Create the skill
        skill = GetState(robot=self.robot)
        
        result = skill()
        
        # Verify expected results
        self.assertTrue(result['success'])
        self.assertEqual(result['position']['x'], 1.0)
        self.assertEqual(result['position']['y'], 2.0)
        self.assertEqual(result['position']['z'], 3.0)
        self.assertEqual(result['orientation']['roll'], 0.1)
        self.assertEqual(result['orientation']['pitch'], 0.2)
        self.assertEqual(result['orientation']['yaw'], 0.3)
        
        # Verify ROS state
        self.assertIn('ros_state', result)
        self.assertEqual(result['ros_state']['battery'], 0.75)
        self.assertEqual(result['ros_state']['status'], "operational")
        
        self.robot.get_pose.assert_called_once()
        self.mock_ros_control.get_state.assert_called_once()
        
    def test_get_state_without_details(self):
        """Test that the GetState skill can exclude detailed ROS state when specified."""
        # Create the skill with include_details=False
        skill = GetState(robot=self.robot, include_details=False)
        
        result = skill()
        
        self.assertTrue(result['success'])
        self.assertEqual(result['position']['x'], 1.0)
        self.assertEqual(result['position']['y'], 2.0)
        self.assertEqual(result['position']['z'], 3.0)
        
        # Verify ROS state is not included
        self.assertNotIn('ros_state', result)
        
        # Verify that only get_pose was called, not ros_control.get_state
        self.robot.get_pose.assert_called_once()
        self.mock_ros_control.get_state.assert_not_called()
        
    def test_get_state_with_missing_pose(self):
        """Test the GetState skill behavior when get_pose raises an exception."""
        # Make get_pose raise an exception
        self.robot.get_pose.side_effect = Exception("No pose available")
        
        skill = GetState(robot=self.robot)
        result = skill()
        
        # Verify success flag is still true, but pose data is missing
        self.assertTrue(result['success'])
        self.assertNotIn('position', result)
        self.assertNotIn('orientation', result)
        self.assertIn('pose_error', result)
        
        # Verify ROS state is still included
        self.assertIn('ros_state', result)
        
    def test_get_state_with_missing_ros_state(self):
        """Test the GetState skill behavior when get_state raises an exception."""
        # Make ros_control.get_state raise an exception
        self.mock_ros_control.get_state.side_effect = Exception("No ROS state available")
        
        skill = GetState(robot=self.robot)
        result = skill()
        
        # Verify success flag is still true and pose data is present
        self.assertTrue(result['success'])
        self.assertIn('position', result)
        self.assertIn('orientation', result)
        
        # Verify ROS state error is included
        self.assertNotIn('ros_state', result)
        self.assertIn('ros_state_error', result)


if __name__ == "__main__":
    unittest.main()
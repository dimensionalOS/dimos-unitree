import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from go2_interfaces.msg import Go2State, IMU
from enum import Enum, auto
import threading
import time
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

from dimos.robot.ros_skill_library import register_skill

__all__ = ['ROSControl', 'RobotMode']

class RobotMode(Enum):
    """Enum for robot modes"""
    UNKNOWN = auto()
    IDLE = auto()
    STANDING = auto()
    MOVING = auto()
    ERROR = auto()

class ROSControl(ABC):
    """Abstract base class for ROS-controlled robots"""
    
    def __init__(self, 
                 node_name: str,
                 cmd_vel_topic: str = 'cmd_vel',
                 max_linear_velocity: float = 1.0,
                 max_angular_velocity: float = 2.0):
        """
        Initialize base ROS control interface
        Args:
            node_name: Name for the ROS node
            cmd_vel_topic: Topic for velocity commands
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        # Initialize ROS node
        if not rclpy.ok():
            rclpy.init()
        
        self._node = Node(node_name)
        self._logger = self._node.get_logger()
        
        # Movement constraints
        self.MAX_LINEAR_VELOCITY = max_linear_velocity
        self.MAX_ANGULAR_VELOCITY = max_angular_velocity
        
        # State tracking
        self._mode = RobotMode.UNKNOWN
        self._is_moving = False
        self._current_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Publishers
        self._cmd_vel_pub = self._node.create_publisher(
            Twist, cmd_vel_topic, 10)
            
        # Start ROS spin thread
        self._spin_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._spin_thread.start()
        
        self._logger.info(f"{node_name} initialized")
    
    def _ros_spin(self):
        """Background thread for ROS spinning"""
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)
    
    def _clamp_velocity(self, velocity: float, max_velocity: float) -> float:
        """Clamp velocity within safe limits"""
        return max(min(velocity, max_velocity), -max_velocity)
    
    @abstractmethod
    def _update_mode(self, *args, **kwargs):
        """Update robot mode based on state - to be implemented by child classes"""
        pass
    
    @register_skill("move_robot")
    def move(self, x: float, y: float, yaw: float, duration: float = 0.0) -> bool:
        """
        Send movement command to the robot
        Args:
            x: Forward/backward velocity (m/s)
            y: Left/right velocity (m/s)
            yaw: Rotational velocity (rad/s)
            duration: How long to move (seconds). If 0, command is continuous
        Returns:
            bool: True if command was sent successfully
        """
        # Clamp velocities to safe limits
        x = self._clamp_velocity(x, self.MAX_LINEAR_VELOCITY)
        y = self._clamp_velocity(y, self.MAX_LINEAR_VELOCITY)
        yaw = self._clamp_velocity(yaw, self.MAX_ANGULAR_VELOCITY)
        
        # Create and send command
        cmd = Twist()
        cmd.linear.x = float(x)
        cmd.linear.y = float(y)
        cmd.angular.z = float(yaw)
        
        try:
            if duration > 0:
                start_time = time.time()
                while time.time() - start_time < duration:
                    self._cmd_vel_pub.publish(cmd)
                    time.sleep(0.1)  # 10Hz update rate
                # Stop after duration
                self.stop()
            else:
                self._cmd_vel_pub.publish(cmd)
            
            self._current_velocity = {"x": x, "y": y, "z": yaw}
            self._is_moving = any(abs(v) > 0.01 for v in [x, y, yaw])
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to send movement command: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop all robot movement
        Returns:
            bool: True if stop command was sent successfully
        """
        try:
            cmd = Twist()
            self._cmd_vel_pub.publish(cmd)
            self._current_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
            self._is_moving = False
            return True
        except Exception as e:
            self._logger.error(f"Failed to send stop command: {e}")
            return False
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current robot state - to be implemented by child classes"""
        pass
    
    def cleanup(self):
        """Cleanup ROS node and stop robot"""
        self.stop()
        self._node.destroy_node()
        rclpy.shutdown()
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from go2_interfaces.msg import Go2State, IMU
from unitree_go.msg import WebRtcReq
from nav2_msgs.action import DriveOnHeading, Spin, BackUp
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from dimos.stream.video_provider import VideoProvider
from enum import Enum, auto
import threading
import time
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)
#from dimos.stream.data_provider import ROSDataProvider
from dimos.stream.ros_video_provider import ROSVideoProvider
import math
from nav2_simple_commander.robot_navigator import BasicNavigator
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point


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
                 webrtc_topic: str = 'webrtc_req',
                 camera_topics: Dict[str, str] = None,
                 use_compressed_video: bool = False,
                 max_linear_velocity: float = 1.0,
                 max_angular_velocity: float = 2.0):
        """
        Initialize base ROS control interface
        Args:
            node_name: Name for the ROS node
            webrtc_topic: Topic for WebRTC commands
            camera_topics: Dictionary of camera topics
            use_compressed_video: Whether to use compressed video
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        # Initialize rclpy and ROS node if not already running
        if not rclpy.ok():
            rclpy.init()

        
        self._node = Node(node_name)
        self._logger = self._node.get_logger()
        
        # Prepare a multi-threaded executor
        self._executor = MultiThreadedExecutor()
        
        # Movement constraints
        self.MAX_LINEAR_VELOCITY = max_linear_velocity
        self.MAX_ANGULAR_VELOCITY = max_angular_velocity
        
        # State tracking
        self._mode = RobotMode.UNKNOWN
        self._is_moving = False
        self._current_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._subscriptions = []


        # Create sensor data QoS profile
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Initialize data handling
        self._video_provider = None
        self._bridge = None
        if camera_topics:
            self._bridge = CvBridge()
            self._video_provider = ROSVideoProvider(dev_name=f"{node_name}_video")
            
            # Create subscribers for each topic with sensor QoS
            msg_type = CompressedImage if use_compressed_video else Image
            for topic in camera_topics.values():
                self._logger.info(f"Subscribing to {topic} with BEST_EFFORT QoS")
                subscription = self._node.create_subscription(
                    msg_type,
                    topic,
                    self._image_callback,
                    sensor_qos
                )
                self._subscriptions.append(subscription)
        
        # Nav2 Action Clients
        self._drive_client = ActionClient(self._node, DriveOnHeading, 'drive_on_heading')
        self._spin_client = ActionClient(self._node, Spin, 'spin')
        self._backup_client = ActionClient(self._node, BackUp, 'backup')
        
        # Wait for action servers
        self._drive_client.wait_for_server()
        self._spin_client.wait_for_server()
        self._backup_client.wait_for_server()

        # Publishers
        self._webrtc_pub = self._node.create_publisher(
            WebRtcReq, webrtc_topic, 10)
            
        # Start ROS spin in a background thread via the executor
        self._spin_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._spin_thread.start()
        
        self._logger.info(f"{node_name} initialized with multi-threaded executor")
    
    def _ros_spin(self):
        """Background thread for spinning the multi-threaded executor."""
        self._executor.add_node(self._node)
        try:
            self._executor.spin()
        finally:
            self._executor.shutdown()
    
    def _clamp_velocity(self, velocity: float, max_velocity: float) -> float:
        """Clamp velocity within safe limits"""
        return max(min(velocity, max_velocity), -max_velocity)
    
    @abstractmethod
    def _update_mode(self, *args, **kwargs):
        """Update robot mode based on state - to be implemented by child classes"""
        pass
    
    def _image_callback(self, msg):
        """Convert ROS image to numpy array and push to data stream"""
        if self._video_provider and self._bridge:
            try:
                if isinstance(msg, CompressedImage):
                    frame = self._bridge.compressed_imgmsg_to_cv2(msg)
                else:
                    frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
                self._video_provider.push_data(frame)
            except Exception as e:
                self._logger.error(f"Error converting image: {e}")
                print(f"Full conversion error: {str(e)}")
    
    @property
    def video_provider(self) -> Optional[ROSVideoProvider]:
        """Data provider property for streaming data"""
        return self._video_provider
    

    def move(self, distance: float, speed: float = 0.5 ,time_allowance: float = 20) -> bool:
        """
        Move the robot forward by a specified distance
        
        Args:
            distance: Distance to move forward in meters (must be positive)
            speed: Speed to move at in m/s (default 0.5)
        Returns:
            bool: True if movement succeeded
        """
        try:
            if distance <= 0:
                self._logger.error("Distance must be positive")
                return False
                
            speed = min(abs(speed), self.MAX_LINEAR_VELOCITY)
            
            # Create DriveOnHeading goal
            goal = DriveOnHeading.Goal()
            goal.target.x = distance
            goal.target.y = 0.0
            goal.target.z = 0.0
            goal.speed = speed
            goal.time_allowance = Duration(sec=time_allowance)
            
            self._logger.info(f"Moving forward: distance={distance}m, speed={speed}m/s")
            
            # Send goal
            goal_future = self._drive_client.send_goal_async(goal)
            goal_future.add_done_callback(self._goal_response_callback)
            
            # Wait for completion
            rclpy.spin_until_future_complete(self._node, goal_future)
            goal_handle = goal_future.result()
            
            if not goal_handle.accepted:
                self._logger.error('DriveOnHeading goal rejected')
                return False
                
            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self._node, result_future)
            
            return True
                
        except Exception as e:
            self._logger.error(f"Forward movement failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return False
            
    def reverse(self, distance: float, speed: float = 0.5, time_allowance: float = 20) -> bool:
        """
        Move the robot backward by a specified distance
        
        Args:
            distance: Distance to move backward in meters (must be positive)
            speed: Speed to move at in m/s (default 0.5)
        Returns:
            bool: True if movement succeeded
        """
        try:
            if distance <= 0:
                self._logger.error("Distance must be positive")
                return False
                
            speed = min(abs(speed), self.MAX_LINEAR_VELOCITY)
            
            # Create BackUp goal
            goal = BackUp.Goal()
            goal.target = Point()
            goal.target.x = -distance  # Negative for backward motion
            goal.target.y = 0.0
            goal.target.z = 0.0
            goal.speed = speed  # BackUp expects positive speed
            goal.time_allowance = Duration(sec=time_allowance)
            
            self._logger.info(f"Moving backward: distance={distance}m, speed={speed}m/s")
            
            # Send goal
            goal_future = self._backup_client.send_goal_async(goal)
            goal_future.add_done_callback(self._goal_response_callback)
            
            # Wait for completion
            rclpy.spin_until_future_complete(self._node, goal_future)
            goal_handle = goal_future.result()
            
            if not goal_handle.accepted:
                self._logger.error('BackUp goal rejected')
                return False
                
            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self._node, result_future)
            
            return True
                
        except Exception as e:
            self._logger.error(f"Backward movement failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return False
            
    def spin(self, degrees: float, speed: float = 45.0, time_allowance: float = 20) -> bool:
        """
        Rotate the robot by a specified angle
        
        Args:
            degrees: Angle to rotate in degrees (positive for counter-clockwise, negative for clockwise)
            speed: Angular speed in degrees/second (default 45.0)
        Returns:
            bool: True if movement succeeded
        """
        try:
            # Convert degrees to radians
            angle = math.radians(degrees)
            angular_speed = math.radians(abs(speed))
            
            # Clamp angular speed
            angular_speed = min(angular_speed, self.MAX_ANGULAR_VELOCITY)
            time_allowance = max(int(abs(angle) / angular_speed * 2), 20)  # At least 20 seconds or double the expected time
            
            # Create Spin goal
            goal = Spin.Goal()
            goal.target_yaw = angle  # Nav2 Spin action expects radians
            goal.time_allowance = Duration(sec=time_allowance)
            
            self._logger.info(f"Spinning: angle={degrees}deg ({angle:.2f}rad)")
            
            # Send goal
            goal_future = self._spin_client.send_goal_async(goal)
            goal_future.add_done_callback(self._goal_response_callback)
            
            # Wait for completion
            rclpy.spin_until_future_complete(self._node, goal_future)
            goal_handle = goal_future.result()
            
            if not goal_handle.accepted:
                self._logger.error('Spin goal rejected')
                return False
                
            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self._node, result_future)
            
            return True
                
        except Exception as e:
            self._logger.error(f"Spin movement failed: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return False
    
    def _goal_response_callback(self, future):
        """Handle the goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._logger.warn('Goal was rejected!')
            return

        self._logger.info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)
    
    def _goal_result_callback(self, future):
        """Handle the goal result."""
        try:
            result = future.result().result
            self._logger.info('Goal completed')
        except Exception as e:
            self._logger.error(f'Goal failed with error: {e}')
    
    def stop(self) -> bool:
        """Stop all robot movement"""
        try:
            self.navigator.cancelTask()
            self._current_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
            self._is_moving = False
            return True
        except Exception as e:
            self._logger.error(f"Failed to stop movement: {e}")
            return False
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current robot state - to be implemented by child classes"""
        pass
    
    def cleanup(self):
        """Cleanup the executor, ROS node, and stop robot."""
        self.stop()

        # Shut down the executor to stop spin loop cleanly
        self._executor.shutdown()

        # Destroy node and shutdown rclpy
        self._node.destroy_node()
        rclpy.shutdown()

    def webrtc_req(self, api_id: int, topic: str = 'rt/api/sport/request', parameter: str = '', priority: int = 0) -> bool:
        """
        Send a WebRTC request command to the robot
        
        Args:
            api_id: The API ID for the command
            topic: The topic to publish to (e.g. 'rt/api/sport/request')
            parameter: Optional parameter string
            priority: Priority level (0 or 1)
            
        Returns:
            bool: True if command was sent successfully
        """
        try:
            # Create and send command
            cmd = WebRtcReq()
            cmd.api_id = api_id
            cmd.topic = topic
            cmd.parameter = parameter
            cmd.priority = priority
            
            self._webrtc_pub.publish(cmd)
            self._logger.info(f"Sent WebRTC request: api_id={api_id}, topic={topic}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to send WebRTC request: {e}")
            return False
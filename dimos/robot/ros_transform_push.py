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

import rclpy
from tf2_ros import Buffer, TransformListener
import tf2_ros
from dimos.utils.logging_config import setup_logger
from reactivex import Observable, Subject
from reactivex import operators as ops
from threading import Lock

logger = setup_logger("dimos.robot.ros_transform")

__all__ = ["ROSTransform"]


class ROSTransformAbility:
    """Base class for handling ROS transforms between coordinate frames"""

    def get_transform_stream(
        self,
        child_frame: str,
        parent_frame: str = "base_link",  # by default we are interested in relating to robot base frame
        frequency: float = 10,  # hz
        timeout: float = 1.0,
    ) -> Observable:
        """
        Creates an Observable stream of transforms between coordinate frames

        Args:
            parent_frame: Parent/source coordinate frame
            child_frame: Child/target coordinate frame
            timeout: How long to wait for the transform to become available (seconds)

        Returns:
            Observable: A stream of TransformStamped messages
        """
        subject = Subject()
        lock = Lock()

        if not hasattr(self, "_tf_buffer"):
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)

        # Create a timer to poll transforms at regular intervals
        def transform_callback():
            with lock:
                try:
                    transform = self._tf_buffer.lookup_transform(
                        parent_frame,
                        child_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=timeout),
                    )
                    subject.on_next(transform)
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    logger.warning(f"Transform lookup failed: {e}")

        # Set up timer in ROS node
        if not hasattr(self, "_node") or self._node is None:
            raise ValueError(
                "ROSTransformAbility requires a ROS node to create transform streams"
            )

        timer = self._node.create_timer(1 / frequency, transform_callback)

        # Return the observable with cleanup
        return subject.pipe(
            ops.finally_action(lambda: self._node.destroy_timer(timer)),
            ops.distinct_until_changed(),
            ops.share(),
        )

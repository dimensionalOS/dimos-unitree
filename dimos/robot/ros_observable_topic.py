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

import functools
import reactivex as rx
from reactivex import operators as ops
from reactivex.disposable import Disposable
from reactivex.scheduler import ThreadPoolScheduler
from rxpy_backpressure import BackPressure

from nav_msgs import msg
from dimos.utils.logging_config import setup_logger
from dimos.utils.threadpool import get_scheduler

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

__all__ = ["ROSObservableTopicAbility"]

# TODO: should go to some shared file, this is copy pasted from ros_control.py
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=1,
)

logger = setup_logger("dimos.robot.ros_control.observable_topic")


class ROSObservableTopicAbility:
    # Ensures that we can return multiple observables which have multiple subscribers
    # consuming the same topic at different (blocking) rates while:
    #
    # - immediately returning latest value received to new subscribers
    # - allowing slow subscribers to consume the topic without blocking fast ones
    # - dealing with backpressure from slow subscribers (auto dropping unprocessed messages)
    #
    # (for more details see corresponding test file)
    #
    # ROS thread ─► ReplaySubject─► observe_on(pool) ─► backpressure.latest ─► sub1 (fast)
    #                          ├──► observe_on(pool) ─► backpressure.latest ─► sub2 (slow)
    #                          └──► observe_on(pool) ─► backpressure.latest ─► sub3 (slower)
    #
    @functools.lru_cache(maxsize=None)
    def topic(
        self,
        topic_name: str,
        msg_type: msg,
        qos=sensor_qos,
        scheduler: ThreadPoolScheduler | None = None,
        drop_unprocessed: bool = True,
    ) -> rx.Observable:
        if scheduler is None:
            scheduler = get_scheduler()

        # upstream ROS callback
        def _on_subscribe(obs, _):
            ros_sub = self._node.create_subscription(msg_type, topic_name, obs.on_next, qos)
            return Disposable(lambda: self._node.destroy_subscription(ros_sub))

        upstream = rx.create(_on_subscribe)

        # hot, latest-cached core
        core = upstream.pipe(
            ops.replay(buffer_size=1),
            ops.ref_count(),  # still synchronous!
        )

        # per-subscriber factory
        def per_sub():
            # hop off the ROS thread into the pool
            base = core.pipe(ops.observe_on(scheduler))

            # optional back-pressure handling
            if not drop_unprocessed:
                return base

            def _subscribe(observer, sch=None):
                return base.subscribe(BackPressure.LATEST(observer), scheduler=sch)

            return rx.create(_subscribe)

        # each `.subscribe()` call gets its own async backpressure chain
        return rx.defer(lambda *_: per_sub())

    # Returns an async function that let's you fetch the latest value from the topic.
    #
    # usage:
    #    costmap = robot.ros_control.topic_value("map", msg.OccupancyGrid, timeout=10)
    #    await costmap()
    #    costmap.dispose()  # clean up the subscription
    #
    # TODO: we could have a diff approach. topic_value is async, it awaits for the first value in the stream.
    # but follow up calls to topic_value would not be async, they would just return the latest value.
    #
    # also we can ensure that it's using self.topic correctly and when all topic_values are disposed sub should
    # stop, and given that first topic_value is initialized (got the first message), second one should be immediate
    # since self.topic if caching the last message.
    #
    # (test if we basically get this behaviour by doing:
    #
    # bla = topic_value()
    # await bla()
    # (follow up awaits should take no time)
    #
    def topic_value(self, topic_name: str, msg_type: msg, qos=sensor_qos, timeout: float | None = None):
        source = self.topic(topic_name, msg_type, qos, latest_only=False)

        # Cache exactly one element and start the ROS subscription immediately.
        connectable = source.pipe(ops.replay(buffer_size=1))
        connection = connectable.connect()  # returns a Disposable

        # Helper that picks off the latest element.
        async def read_value():
            obs = connectable.pipe(ops.take(1))
            if timeout is not None:
                obs = obs.pipe(ops.timeout(timeout))
            return await obs.to_future()

        # Ensure the caller can clean up after itself.
        read_value.dispose = lambda: connection.dispose()

        return read_value

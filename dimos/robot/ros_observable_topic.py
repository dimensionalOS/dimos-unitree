import functools
import reactivex as rx
from reactivex import operators as ops
from reactivex.disposable import Disposable
from reactivex.observable import Observable
from reactivex.scheduler import ThreadPoolScheduler

from nav_msgs import msg
from dimos.utils.logging_config import setup_logger
from dimos.utils.threadpool import get_scheduler


from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

# TODO: should go to some shared file, this is copy pasted from ros_control.py
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=1,
)

logger = setup_logger("dimos.robot.ros_control.observable_topic")


class ROSObservableTopicAbility:
    # multiple systems can observe the topic, we keep a single sub callback from rxpy/ros
    # PERF: we might want a monitorable count for this cache
    @functools.lru_cache(maxsize=None)
    def topic(
        self,
        topic_name: str,
        msg_type: msg,
        qos=sensor_qos,
        cold_sub=False,
        scheduler: ThreadPoolScheduler = None,
    ) -> Observable:
        logger.info(f"Subscribing to {topic_name} with {sensor_qos} QoS")

        if not scheduler:
            scheduler = get_scheduler()

        def _on_subscribe(observer, scheduler):
            # sub to ros
            ros_sub = self._node.create_subscription(
                msg_type, topic_name, observer.on_next, qos
            )

            def cleanup():
                self._node.destroy_subscription(ros_sub)

            # ensure we are unsubing when we should be
            return Disposable(cleanup)

        observable = rx.create(_on_subscribe).pipe(
            ops.observe_on(scheduler)  # hop off ROS callback thread asap
        )

        if not cold_sub:
            return observable.pipe(
                ops.replay(buffer_size=1), ops.ref_count()
            ).observe_on(scheduler)

        return observable.pipe(ops.publish(), ops.ref_count()).observe_on(scheduler)

    @functools.lru_cache(maxsize=None)
    def topic2(
        self,
        topic_name: str,
        msg_type: msg,
        qos=sensor_qos,
        scheduler: ThreadPoolScheduler | None = None,
        drop_unprocessed: bool = True,
        sample_period: float = 0.02,  # 50 Hz default
    ) -> rx.Observable:
        """
        Return a hot observable of `msg_type` that:
          • hops off the ROS callback thread immediately,
          • replays the last value to new subscribers,
          • (optionally) sheds overload by emitting only the latest sample.

        Set `drop_unprocessed=False` if you *really* need every frame.
        """

        # 1. Choose where downstream work will run
        if scheduler is None:
            scheduler = ThreadPoolScheduler()  # or your custom pool

        def _on_subscribe(observer, _):
            ros_sub = self._node.create_subscription(
                msg_type, topic_name, observer.on_next, qos
            )
            return Disposable(lambda: self._node.destroy_subscription(ros_sub))

        # 2. Hop off ROS thread ASAP
        source = rx.create(_on_subscribe).pipe(ops.observe_on(scheduler))

        # 3. Apply the protective back‑pressure wrapper if requested
        if drop_unprocessed:
            source = source.pipe(ops.sample(sample_period))

        # 4. Turn it into a hot, shared, “latest‑value” observable
        return source.pipe(
            ops.replay(buffer_size=1),
            ops.ref_count(),
        )

# file: dimos/stream/ros_video_provider.py

from reactivex import Subject, Observable
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
import multiprocessing
import logging

from dimos.stream.video_provider import AbstractVideoProvider

logging.basicConfig(level=logging.INFO)

# Optional: if you want concurrency or time-based sampling
pool_scheduler = ThreadPoolScheduler(multiprocessing.cpu_count())

class ROSVideoProvider(AbstractVideoProvider):
    """Video provider that uses a Subject to broadcast frames pushed by ROS."""

    def __init__(self, dev_name: str = "ros_video"):
        super().__init__(dev_name)
        self.logger = logging.getLogger(dev_name)
        # This subject will receive frames
        self._subject = Subject()

    def push_data(self, frame):
        """Push a new frame into the provider."""
        print(f"ROSVideoProvider pushing frame of type {type(frame)}")
        self.logger.debug(f"ROSVideoProvider pushing frame of type {type(frame)}")
        self._subject.on_next(frame)

    def capture_video_as_observable(self, fps: int = 30) -> Observable:
        """
        Return an observable of frames. If you want to do time-based sampling,
        you can apply it here (e.g. ops.sample).
        """
        base_stream = self._subject.pipe(
            ops.observe_on(pool_scheduler),
            # share() so multiple subscribers can consume the same frames
            ops.share()
        )

        # If you want to sample to a maximum of `fps` frames per second,
        # you can do so here. Otherwise, just return base_stream.
        if fps and fps > 0:
            return base_stream.pipe(
                ops.sample(1.0 / fps, scheduler=pool_scheduler),
                ops.share()
            )
        else:
            return base_stream

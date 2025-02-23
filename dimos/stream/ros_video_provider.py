# file: dimos/stream/ros_video_provider.py

from reactivex import Subject, Observable
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
import multiprocessing
import logging
import time

from dimos.stream.video_provider import AbstractVideoProvider

logging.basicConfig(level=logging.INFO)

# Create thread pool scheduler
pool_scheduler = ThreadPoolScheduler(multiprocessing.cpu_count())

class ROSVideoProvider(AbstractVideoProvider):
    """Video provider that uses a Subject to broadcast frames pushed by ROS."""

    def __init__(self, dev_name: str = "ros_video"):
        super().__init__(dev_name)
        self.logger = logging.getLogger(dev_name)
        self._subject = Subject()
        self._last_frame_time = None
        print("ROSVideoProvider initialized")

    def push_data(self, frame):
        """Push a new frame into the provider."""
        try:
            current_time = time.time()
            if self._last_frame_time:
                frame_interval = current_time - self._last_frame_time
                print(f"Frame interval: {frame_interval:.3f}s ({1/frame_interval:.1f} FPS)")
            self._last_frame_time = current_time
            
            print(f"Pushing frame type: {type(frame)}")
            self._subject.on_next(frame)
            print("Frame pushed")
        except Exception as e:
            print(f"Push error: {e}")

    def capture_video_as_observable(self, fps: int = 30) -> Observable:
        """Return an observable of frames."""
        print(f"Creating observable (fps={fps})")
        
        # Create base pipeline without rate limiting first
        base_pipeline = self._subject.pipe(
            ops.do_action(lambda x: print("BASE: Got frame")),
            ops.share()
        )
        
        # Add debug subscription to base pipeline
        base_pipeline.subscribe(
            on_next=lambda x: print("BASE SUB: Frame received"),
            on_error=lambda e: print(f"BASE SUB: Error: {e}")
        )
        
        # If fps specified, add rate limiting
        if fps and fps > 0:
            print(f"Adding rate limiting at {fps} FPS")
            rate_limited = base_pipeline.pipe(
                # Use throttle_first instead of sample for inconsistent sources
                ops.throttle_first(1.0 / fps, scheduler=pool_scheduler),
                ops.do_action(lambda x: print(f"RATE LIMITED: Frame passed throttle")),
                ops.share()
            )
            
            # Debug subscription for rate-limited pipeline
            rate_limited.subscribe(
                on_next=lambda x: print("RATE SUB: Frame received"),
                on_error=lambda e: print(f"RATE SUB: Error: {e}")
            )
            
            return rate_limited
        
        return base_pipeline

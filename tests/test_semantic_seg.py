import cv2
import numpy as np
import os
import sys
import queue
import threading

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimos.stream.video_provider import VideoProvider
from dimos.perception.semantic_seg import SemanticSegmentationStream

def main():
    # Create a queue for thread communication (limit to prevent memory issues)
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    
    # Logitech C920e camera parameters at 480p
    camera_params = {
        'resolution': (640, 480),  # 480p resolution
        'focal_length': 3.67,  # mm
        'sensor_size': (4.8, 3.6)  # mm (1/4" sensor)
    }
    
    # Initialize video provider and segmentation stream
    video_provider = VideoProvider("test_camera", video_source=0)
    seg_stream = SemanticSegmentationStream(enable_depth=True, camera_params=camera_params)
    
    # Create streams
    video_stream = video_provider.capture_video_as_observable(realtime=False, fps=2)
    segmentation_stream = seg_stream.create_stream(video_stream)
    
    # Define callbacks for the segmentation stream
    def on_next(segmentation):
        if stop_event.is_set():
            return
            
        # Get the frame and visualize
        vis_frame = segmentation.metadata["viz_frame"]

        # Put frame in queue for main thread to display (non-blocking)
        try:
            frame_queue.put_nowait(vis_frame)
        except queue.Full:
            # Skip frame if queue is full
            pass
    
    def on_error(error):
        print(f"Error: {error}")
        stop_event.set()
    
    def on_completed():
        print("Stream completed")
        stop_event.set()
    
    # Start the subscription
    subscription = None
    
    try:
        # Subscribe to start processing in background thread
        subscription = segmentation_stream.subscribe(
            on_next=on_next,
            on_error=on_error,
            on_completed=on_completed
        )
        
        print("Semantic segmentation visualization started. Press 'q' to exit.")
        
        # Main thread loop for displaying frames
        while not stop_event.is_set():
            try:
                # Get frame with timeout (allows checking stop_event periodically)
                vis_frame = frame_queue.get(timeout=1.0)
                
                # Display the frame in main thread
                cv2.imshow("Semantic Segmentation", vis_frame)
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key pressed")
                    break
                    
            except queue.Empty:
                # No frame available, check if we should continue
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key pressed")
                    break
                continue
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping...")
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Clean up resources
        if subscription:
            subscription.dispose()
        
        video_provider.dispose_all()
        seg_stream.cleanup()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
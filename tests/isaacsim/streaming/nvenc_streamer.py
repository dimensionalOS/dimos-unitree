import numpy as np
import subprocess
import queue
import threading
import time
import cv2

class NVENCStreamer:
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.encoder_thread = None
        self.frames_processed = 0
        self.start_time = None
        
        # FFmpeg command using RTSP output with more robust settings
        self.ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgba',  # Changed from bgr24 to rgba since Isaac Sim outputs RGBA
            '-r', str(fps),
            '-i', '-',  # Read from pipe
            '-c:v', 'h264_nvenc',  # Use NVIDIA encoder
            '-preset', 'p1',  # Low-latency preset
            '-tune', 'ull',  # Ultra-low latency tuning
            '-zerolatency', '1',
            '-b:v', '5M',  # Bitrate
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # Use TCP for more reliable streaming
            'rtsp://localhost:8554/live'  # Stream to local RTSP server first
        ]

    def start(self):
        """Start the encoder thread"""
        if self.running:
            return
        self.running = True
        self.encoder_thread = threading.Thread(target=self._encoder_loop)
        self.encoder_thread.start()
        print("[NVENCStreamer] Encoder thread started")

    def stop(self):
        """Stop the encoder thread"""
        print("[NVENCStreamer] Stopping encoder...")
        self.running = False
        if self.encoder_thread:
            self.encoder_thread.join()
        print("[NVENCStreamer] Encoder stopped")
            
    def push_frame(self, frame: np.ndarray):
        """Push a new frame to the encoding queue with rate limiting"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_frame_time < self.frame_interval:
            return
            
        try:
            # No need to convert RGBA to BGR anymore - we'll use the RGBA directly
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put_nowait(frame)  # Put the RGBA frame directly
            self.last_frame_time = current_time
            
        except Exception as e:
            print(f"[NVENCStreamer] Error processing frame: {str(e)}")

    def _encoder_loop(self):
        if self.start_time is None:
            self.start_time = time.time()
            
        process = subprocess.Popen(
            self.ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False
        )
        
        last_fps_print = time.time()
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
                self.frames_processed += 1
                
                # Print FPS every 5 seconds
                current_time = time.time()
                if current_time - last_fps_print >= 5.0:
                    elapsed = current_time - self.start_time
                    fps = self.frames_processed / elapsed
                    print(f"[NVENCStreamer] Streaming at {fps:.2f} FPS")
                    last_fps_print = current_time
                    
            except queue.Empty:
                continue
            except BrokenPipeError:
                print("[NVENCStreamer] FFmpeg process closed unexpectedly")
                break
            except Exception as e:
                print(f"[NVENCStreamer] Streaming error: {str(e)}")
                break
                
        process.stdin.close()
        process.wait() 
import numpy as np
import subprocess
import queue
import threading
import time
import cv2

class NVENCStreamer:
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 60):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self.last_fps_print = time.time()
        self.frame_queue = queue.Queue(maxsize=120)
        self.running = False
        self.encoder_thread = None
        self.frames_processed = 0
        self.start_time = None
        
        # Update FFmpeg command to match the standalone implementation
        self.ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-loglevel', 'debug',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            '-r', str(fps),
            '-i', '-',
            '-an',  # No audio
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-f', 'rtsp',
            'rtsp://mediamtx:8554/stream',
            '-rtsp_transport', 'tcp'
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
            # Convert RGBA to BGR (if input is RGBA)
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put_nowait(frame)
            self.last_frame_time = current_time
            
        except Exception as e:
            print(f"[NVENCStreamer] Error processing frame: {str(e)}")

    def _encoder_loop(self):
        """Encoder loop that matches the standalone implementation"""
        if self.start_time is None:
            self.start_time = time.time()
            
        try:
            process = subprocess.Popen(
                self.ffmpeg_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False
            )
            
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    # Write frame directly to stdin, matching standalone implementation
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                    self.frames_processed += 1
                    
                    current_time = time.time()
                    if current_time - self.last_fps_print >= 5.0:
                        elapsed = current_time - self.start_time
                        fps = self.frames_processed / elapsed
                        print(f"[NVENCStreamer] Streaming at {fps:.2f} FPS")
                        self.last_fps_print = current_time
                        
                except queue.Empty:
                    continue
                except BrokenPipeError:
                    print("[NVENCStreamer] Broken pipe, stopping...")
                    break
                except Exception as e:
                    print(f"[NVENCStreamer] Streaming error: {str(e)}")
                    break
                
        except Exception as e:
            print(f"[NVENCStreamer] Process error: {str(e)}")
        finally:
            if process:
                process.stdin.close()
                process.wait() 
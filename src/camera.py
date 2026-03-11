import numpy as np
import cv2
import config

try:
    from picamera2 import Picamera2
    HAS_PICAM = True
except ImportError:
    HAS_PICAM = False

class CameraHandler:
    """Wraps picamera2 for frame capture, with OpenCV fallback for Mac/PC."""
    
    def __init__(self, source=None):
        self.source = source or config.VIDEO_SOURCE
        
        if self.source:
            self.mode = "video"
            print(f"Initializing Video Source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                 print(f"Error: Could not open video file {self.source}")
        elif HAS_PICAM:
            self.mode = "pi"
            self.picam2 = Picamera2()
            config_pi = self.picam2.create_video_configuration(main={"size": (640, 480)})
            self.picam2.configure(config_pi)
            self.picam2.start()
        else:
            self.mode = "webcam"
            print("Warning: Picamera2 not found. Initializing MacOS/OpenCV Webcam fallback...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam at index 0. Trying index 1...")
                self.cap = cv2.VideoCapture(1)
            
            if self.cap.isOpened():
                import time
                time.sleep(1.0) # Warm-up
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                print("Error: Could not open any webcam.")

    def capture_frame(self):
        """
        Captures a frame and returns it as a uint8 BGR numpy array.
        """
        if self.mode == "pi":
            frame = self.picam2.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                return np.zeros((480, 640, 3), dtype=np.uint8)

            ret, frame = self.cap.read()
            
            # Loop video if it ends
            if not ret and self.mode == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            if not ret:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            return frame

    def cleanup(self):
        if self.mode == "webcam" and hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    cam = CameraHandler()
    frame = cam.capture_frame()
    print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
    cam.cleanup()

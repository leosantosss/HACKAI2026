import numpy as np
import cv2

try:
    from picamera2 import Picamera2
    HAS_PICAM = True
except ImportError:
    HAS_PICAM = False

class CameraHandler:
    """Wraps picamera2 for frame capture, with OpenCV fallback for Mac/PC."""
    
    def __init__(self):
        if HAS_PICAM:
            self.mode = "pi"
            self.picam2 = Picamera2()
            # Standardizing to (320, 240) as per spec
            config = self.picam2.create_video_configuration(main={"size": (320, 240)})
            self.picam2.configure(config)
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
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                print("Error: Could not open any webcam.")

    def capture_frame(self):
        """
        Captures a frame and returns it as a (240, 320, 3) uint8 RGB numpy array.
        Note: OpenCV returns BGR, Picamera2 returns RGB by default in capture_array.
        """
        if self.mode == "pi":
            # picamera2 capture_array is usually RGB
            return self.picam2.capture_array()
        else:
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                return np.zeros((240, 320, 3), dtype=np.uint8)

            ret, frame = self.cap.read()
            if not ret:
                return np.zeros((240, 320, 3), dtype=np.uint8)
            
            frame_resized = cv2.resize(frame, (320, 240))
            # Convert BGR to RGB for consistency with Picamera2
            return cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    def cleanup(self):
        if self.mode == "webcam" and hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    cam = CameraHandler()
    frame = cam.capture_frame()
    print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
    cam.cleanup()

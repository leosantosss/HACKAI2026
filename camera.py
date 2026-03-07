import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    # Fallback for non-Pi environments (testing)
    class Picamera2:
        def __init__(self):
            print("Warning: Picamera2 not found. Using mock camera.")
        def configure(self, config): pass
        def start(self): pass
        def capture_array(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)
        def create_video_configuration(self, **kwargs):
            class MockConfig:
                def __init__(self): self.main = type('obj', (object,), {'size': (320, 240)})
            return MockConfig()

class CameraHandler:
    """Wraps picamera2 for frame capture."""
    
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (320, 240)})
        self.picam2.configure(config)
        self.picam2.start()

    def capture_frame(self):
        """
        Captures a frame and returns it as a (240, 320, 3) numpy array.
        """
        # capture_array returns the frame in the format configured (RGB)
        return self.picam2.capture_array()

if __name__ == "__main__":
    cam = CameraHandler()
    frame = cam.capture_frame()
    print(f"Captured frame shape: {frame.shape}")

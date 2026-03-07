import time
from camera import CameraHandler
from detector import ObjectDetector
from vibration import BuzzerController
import config

def select_pattern(detection):
    """
    Selects the beep pattern based on detection class and proximity.
    Patterns:
    - pulse_1: person far
    - pulse_2: person medium / bicycle / obstacle
    - rapid_3: person close / vehicle
    - continuous: critical / staircase (proxy here as very close or specific class)
    """
    cls = detection['class']
    ratio = detection['area_ratio']
    
    # Logic for pattern selection
    if ratio > config.PROXIMITY_CLOSE:
        return "continuous" if cls in ["car", "bus", "truck"] else "rapid_3"
    
    if cls == "person":
        if ratio > config.PROXIMITY_MEDIUM:
            return "rapid_3"
        elif ratio > config.PROXIMITY_FAR:
            return "pulse_2"
        else:
            return "pulse_1"
            
    # Vehicles (car, truck, etc.)
    if cls in ["car", "motorcycle", "bus", "truck", "bicycle"]:
        if ratio > config.PROXIMITY_MEDIUM:
            return "rapid_3"
        else:
            return "pulse_2"
            
    # Default for other danger classes (animals, etc.)
    if ratio > config.PROXIMITY_MEDIUM:
        return "pulse_2"
    else:
        return "pulse_1"

def main():
    print("VisionAid: Initializing...")
    try:
        camera = CameraHandler()
        detector = ObjectDetector()
        buzzer = BuzzerController()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("VisionAid: System Ready.")
    
    # track cooldowns per class
    cooldowns = {}
    
    try:
        while True:
            frame = camera.capture_frame()
            detections = detector.detect(frame)
            
            if detections:
                # Detector returns detections sorted by area_ratio (closest first)
                for det in detections:
                    cls = det['class']
                    current_time = time.time()
                    
                    # Check cooldown
                    if cls not in cooldowns or (current_time - cooldowns[cls]) > config.COOLDOWN_TIME:
                        pattern = select_pattern(det)
                        print(f"Action: {pattern} for {cls} (ratio: {det['area_ratio']:.2f})")
                        buzzer.trigger(pattern)
                        cooldowns[cls] = current_time
                        
                        # Only handle the highest priority (closest) detection per frame
                        break
            
            # Small sleep to yield CPU if needed, though camera capture/inference 
            # will likely be the bottleneck.
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nVisionAid: Stopping...")
    finally:
        buzzer.cleanup()

if __name__ == "__main__":
    main()

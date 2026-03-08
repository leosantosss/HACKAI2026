import time
import cv2
from camera import CameraHandler
from detector import VisionAgent
from vibration import BuzzerController
import config

def get_cooldown(alert_type):
    """Maps alert type to its cooldown constant."""
    if alert_type == "approach": return config.COOLDOWN_APPROACH
    if alert_type == "obstacle": return config.COOLDOWN_OBSTACLE
    if alert_type == "stair": return config.COOLDOWN_STAIR
    if alert_type == "curb": return config.COOLDOWN_CURB
    return 1.0

def main():
    print("Vizzion: Initializing Refined Pipeline...")
    try:
        camera = CameraHandler()
        agent = VisionAgent()
        buzzer = BuzzerController()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("Vizzion: System Ready.")
    
    # Cooldown tracking: {alert_type: last_triggered_timestamp}
    cooldowns = {}
    
    # Performance and Caching
    frame_count = 0
    start_time = time.time()
    last_alert = None
    last_mask = None
    last_color_mask = None
    last_detections = []

    try:
        while True:
            frame = camera.capture_frame()
            frame_count += 1
            
            # 1. Run inference on a subset of frames (Smarter Frame Skipping)
            if frame_count % config.FRAME_SKIP == 0 or last_mask is None:
                last_alert, debug_data = agent.analyze(frame)
                last_mask, last_detections = debug_data
                if last_mask is not None:
                    last_color_mask = agent.seg_engine.colorize_mask(last_mask)
            
            # 2. Use results from latest AI pass
            alert = last_alert
            mask = last_mask
            color_mask = last_color_mask
            raw_detections = last_detections
            
            if alert:
                alert_type = alert['type']
                current_time = time.time()
                
                # Check cooldown
                cooldown_limit = get_cooldown(alert_type)
                last_time = cooldowns.get(alert_type, 0)
                
                if (current_time - last_time) > cooldown_limit:
                    # Determine intensity for buzzer
                    intensity = 1.0
                    if alert_type == "approach":
                        intensity = alert['growth_rate']
                    elif alert_type == "obstacle":
                        intensity = alert['area_ratio']
                    elif alert_type in ["stair", "curb"]:
                        intensity = alert['confidence']

                    print(f"[{time.strftime('%H:%M:%S')}] ALERT: {alert_type.upper()} | Intensity: {intensity:.2f}")
                    buzzer.trigger(alert_type, intensity)
                    cooldowns[alert_type] = current_time
            
            # 3. Visual Debugging (Always render at full speed)
            if config.SHOW_DISPLAY and color_mask is not None:
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Overlay Segmentation Mask (cached)
                display_frame = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
                
                # 2. Draw raw detections (bboxes) from cache
                for det in raw_detections:
                    left, top, right, bottom = det['bbox']
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 1)
                    cv2.putText(display_frame, f"{det['class']} {det['confidence']:.2f}", 
                                (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 3. Draw approach zone (vertical lines)
                h, w = display_frame.shape[:2]
                margin = int((1.0 - config.APPROACH_ZONE) / 2 * w)
                cv2.line(display_frame, (margin, 0), (margin, h), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.line(display_frame, (w - margin, 0), (w - margin, h), (255, 255, 0), 1, cv2.LINE_AA)

                # 4. Draw Hazard Zone (RED BOX - Bottom Center)
                hazard_h = int(h * config.HAZARD_ZONE_HEIGHT)
                cv2.rectangle(display_frame, (margin, h - hazard_h), (w - margin, h), (0, 0, 255), 2)

                # 5. Overlay FPS and Metadata
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 100, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if alert:
                    msg = f"ALERT: {alert['type'].upper()}"
                    cv2.putText(display_frame, msg, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Convert RGB back to BGR for OpenCV display
                cv2.imshow("Vizzion Monitor", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # Limit loop rate slightly to save CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nVizzion: Stopping...")
    finally:
        buzzer.cleanup()
        camera.cleanup()
        if config.SHOW_DISPLAY:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

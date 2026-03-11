import time
import cv2
import numpy as np
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
    
    # Timing
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = camera.capture_frame()
            frame_count += 1
            
            # 1. Analyze every single frame (Mac Pro Mode)
            alert, debug_data = agent.analyze(frame)
            mask, raw_detections = debug_data
            if mask is not None:
                color_mask = agent.seg_engine.colorize_mask(mask)
            
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
            
            # 3. Visual Debugging
            if config.SHOW_DISPLAY and color_mask is not None:
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # A. Brightness Boost (Gamma Correction)
                gamma = 1.2
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                bright_frame = cv2.LUT(frame, table)

                # B. Blending
                display_frame = cv2.addWeighted(bright_frame, 0.7, color_mask, 0.3, 0)
                h_d, w_d = display_frame.shape[:2]

                # 2. Draw raw detections (bboxes)
                for det in raw_detections:
                    left, top, right, bottom = det['bbox']
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 1)
                    cv2.putText(display_frame, f"{det['class']}", 
                                (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 3. Draw approach zone (vertical lines)
                margin = int((1.0 - config.APPROACH_ZONE) / 2 * w_d)
                cv2.line(display_frame, (margin, 0), (margin, h_d), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.line(display_frame, (w_d - margin, 0), (w_d - margin, h_d), (255, 255, 0), 1, cv2.LINE_AA)

                # 4. Draw Hazard Zone (RED BOX)
                hazard_h = int(h_d * config.HAZARD_ZONE_HEIGHT)
                cv2.rectangle(display_frame, (margin, h_d - hazard_h), (w_d - margin, h_d), (0, 0, 255), 2)

                # 5. Overlay FPS and Metadata
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (w_d - 100, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if alert:
                    msg = f"ALERT: {alert['type'].upper()}"
                    cv2.putText(display_frame, msg, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Direct Display
                cv2.imshow("Vizzion Monitor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # Run as fast as hardware allows
            pass

    except KeyboardInterrupt:
        print("\nVizzion: Stopping...")
    finally:
        buzzer.cleanup()
        camera.cleanup()
        if config.SHOW_DISPLAY:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

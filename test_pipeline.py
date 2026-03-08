"""
test_pipeline.py — Vizzion Pipeline Test
Runs the FULL VisionAgent analysis on the parquet dataset images.
This verifies that the alerts (curb, stair, person, etc.) work with 
the new fine-tuned model.
"""
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from main import VisionAgent
import config

def test_on_dataset():
    print("Initializing VisionAgent with fine-tuned model...")
    agent = VisionAgent()
    
    PARQUET_FILE = "train-00000-of-00001.parquet"
    print(f"Reading {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    
    print("\nStarting Pipeline Test.")
    print("Press any key for next image, 'q' to quit.\n")

    for i, row in df.iterrows():
        # 1. Extract Image
        img_data = row.get('pixel_values') or row.get('image')
        if isinstance(img_data, dict):
            raw = img_data.get('bytes')
            pil_img = Image.open(io.BytesIO(raw)).convert('RGB')
        else:
            pil_img = Image.fromarray(np.array(img_data)).convert('RGB')
        
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 2. Analyze using full pipeline
        alert, debug_data = agent.analyze(frame)
        mask, raw_detections = debug_data
        
        # 3. Visual Debugging (Same as main.py)
        color_mask = agent.seg_engine.colorize_mask(mask)
        display_frame = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
        
        # Draw bboxes
        for det in raw_detections:
            left, top, right, bottom = det['bbox']
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.putText(display_frame, f"{det['class']} {det['confidence']:.2f}", 
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw Approach Zone & Hazard Zone
        h, w = display_frame.shape[:2]
        margin = int((1.0 - config.APPROACH_ZONE) / 2 * w)
        hazard_h = int(h * config.HAZARD_ZONE_HEIGHT)
        cv2.line(display_frame, (margin, 0), (margin, h), (255, 255, 0), 1)
        cv2.line(display_frame, (w - margin, 0), (w - margin, h), (255, 255, 0), 1)
        cv2.rectangle(display_frame, (margin, h - hazard_h), (w - margin, h), (0, 0, 255), 2)

        # Overlay Alert
        if alert:
            msg = f"ALERT: {alert['type'].upper()} ({alert.get('confidence', 0):.2f})"
            cv2.putText(display_frame, msg, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"[{i+1}] ALERT DETECTED: {alert['type']} | Intensity: {alert.get('confidence', 0):.2f}")
        else:
            cv2.putText(display_frame, "PATH CLEAR", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Vizzion Pipeline Test", display_frame)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == "__main__":
    test_on_dataset()

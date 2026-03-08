import cv2
import numpy as np
import config
from collections import deque

class MobileNetDetector:
    """Internal helper to load and run MobileNet SSD v2."""
    def __init__(self):
        self.net = cv2.dnn.readNetFromTensorflow(config.MODEL_WEIGHTS, config.MODEL_PROTO)
        self.classes = {
            1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck",
            10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench"
        }

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        detections = []
        margin = (1.0 - config.APPROACH_ZONE) / 2
        
        for i in range(output.shape[2]):
            confidence = output[0, 0, i, 2]
            
            # Use safety threshold for center-zone objects
            # We first calculate center_x roughly to decide which threshold to use
            temp_left = output[0, 0, i, 3]
            temp_right = output[0, 0, i, 5]
            center_x = (temp_left + temp_right) / 2
            
            # If in center zone, use lower threshold
            is_in_path = margin <= center_x <= 1.0 - margin
            threshold = config.SAFETY_CONFIDENCE_THRESHOLD if is_in_path else config.CONFIDENCE_THRESHOLD
            
            if confidence > threshold:
                class_id = int(output[0, 0, i, 1])
                class_name = self.classes.get(class_id, "unknown")
                
                left = int(output[0, 0, i, 3] * w)
                top = int(output[0, 0, i, 4] * h)
                right = int(output[0, 0, i, 5] * w)
                bottom = int(output[0, 0, i, 6] * h)
                
                # Clip to frame
                left, top = max(0, left), max(0, top)
                right, bottom = min(w, right), min(h, bottom)
                
                area_ratio = ((right - left) * (bottom - top)) / (w * h)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (left, top, right, bottom),
                    'area_ratio': area_ratio,
                    'center_x': center_x
                })
        
        # EMERGENCY BACKUP: If zero detections, check if the camera is virtually "blinded"
        # by a very close object (massive drop in edge density compared to a typical scene).
        if not detections:
            # Analyze focus/blurryness
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Center ROI
            roi_h, roi_w = int(h*0.4), int(w*0.4)
            y1, x1 = int(h*0.3), int(w*0.3)
            roi = gray[y1:y1+roi_h, x1:x1+roi_w]
            
            # Simple check: if the center is very uniform but not black, someone is blocking it
            std_dev = np.std(roi)
            avg_val = np.mean(roi)
            
            # If the image is very uniform (low std dev) but bright enough to not be "night"
            # and the AI found nothing, it's likely a hand/chest right in front of the lens.
            if std_dev < 15 and 40 < avg_val < 250:
                detections.append({
                    'class': 'unknown',
                    'confidence': 0.99, # Force it
                    'bbox': (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)),
                    'area_ratio': 0.8,
                    'center_x': 0.5
                })
                
        return detections

class ApproachDetector:
    """Detects objects rapidly moving toward the user."""
    def __init__(self):
        # history: {track_name: deque of area_ratios}
        self.history = {}
        # emas: {track_name: current_ema_value}
        self.emas = {}

    def analyze(self, detections):
        alerts = []
        current_classes = set()
        
        for det in detections:
            cls = det['class']
            
            # If not in approach classes, check if we allow unknown objects in path
            if cls not in config.APPROACH_CLASSES:
                if not (config.INCLUDE_UNKNOWN_OBJECTS and cls == "unknown"):
                    continue
            
            # Ignore far-away noise if below min size
            if det['area_ratio'] < config.APPROACH_MIN_SIZE:
                continue
            
            # Check center zone
            margin = (1.0 - config.APPROACH_ZONE) / 2
            if not (margin <= det['center_x'] <= 1.0 - margin):
                continue
                
            track_name = cls
            
            # Apply EMA Smoothing to the area_ratio
            alpha = config.APPROACH_EMA_ALPHA
            prev_ema = self.emas.get(track_name, det['area_ratio'])
            current_ema = (alpha * det['area_ratio']) + ((1 - alpha) * prev_ema)
            self.emas[track_name] = current_ema
            
            if track_name not in self.history:
                self.history[track_name] = deque(maxlen=config.APPROACH_TREND_FRAMES)
            
            hist = self.history[track_name]
            hist.append(current_ema)
            current_classes.add(track_name)

            # Need enough history to see a trend
            if len(hist) >= config.APPROACH_TREND_FRAMES:
                # Check for net growth over the window
                net_growth = (hist[-1] - hist[0]) / hist[0] if hist[0] > 0 else 0
                
                # Check consistency: image must be generally growing
                growing_steps = 0
                for i in range(1, len(hist)):
                    if hist[i] > hist[i-1]:
                        growing_steps += 1
                
                # Require 80% of frames in the window to be growing
                is_trending = (growing_steps >= (len(hist) * 0.8)) and (net_growth > 0)
                
                # Average growth per frame in this window
                avg_growth = net_growth / (len(hist) - 1)
                
                if is_trending and avg_growth > config.APPROACH_SENSITIVITY:
                    alerts.append({
                        'type': 'approach',
                        'class': cls,
                        'growth_rate': avg_growth,
                        'area_ratio': current_ema
                    })

        # Clean up history for objects no longer present
        for cls in list(self.history.keys()):
            if cls not in current_classes:
                self.history.pop(cls)
                self.emas.pop(cls, None)
                
        return alerts

class StaticObstacleDetector:
    """Detects obstacles that are in the way but not rapidly approaching."""
    def __init__(self):
        # counters: {track_name: consecutive_growth_frames}
        self.counters = {}
        self.prev_areas = {}

    def analyze(self, detections):
        alerts = []
        current_classes = set()
        
        for det in detections:
            cls = det['class']
            
            # Check class lists
            if cls not in config.OBSTACLE_CLASSES:
                if not (config.INCLUDE_UNKNOWN_OBJECTS and cls == "unknown"):
                    continue
            
            if det['area_ratio'] < config.OBSTACLE_MIN_SIZE:
                continue

            # Check center zone
            margin = (1.0 - config.APPROACH_ZONE) / 2
            if not (margin <= det['center_x'] <= 1.0 - margin):
                continue
            
            track_name = cls
            prev_area = self.prev_areas.get(track_name, 0)
            is_growing = det['area_ratio'] > prev_area
            is_slow = (det['area_ratio'] - prev_area) / prev_area <= config.APPROACH_SENSITIVITY if prev_area > 0 else True
            
            if is_growing and is_slow:
                self.counters[track_name] = self.counters.get(track_name, 0) + 1
            else:
                self.counters[track_name] = 0
            
            if self.counters[track_name] >= config.OBSTACLE_GROWTH_THRESHOLD:
                alerts.append({
                    'type': 'obstacle',
                    'class': cls,
                    'area_ratio': det['area_ratio']
                })
            
            self.prev_areas[track_name] = det['area_ratio']
            current_classes.add(track_name)
            
        # Clean up
        for cls in list(self.counters.keys()):
            if cls not in current_classes:
                self.counters.pop(cls)
                self.prev_areas.pop(cls, None)
                
        return alerts

class StairDetector:
    """Analyzes bottom 1/3 of frame for dense horizontal edges (stair lip)."""
    def analyze(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(2/3*h):h, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate horizontal edge density
        # We look for horizontal lines by convolving with a horizontal kernel or just counting pixels
        density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        
        if density > config.STAIR_CONFIDENCE_THRESHOLD:
            return [{'type': 'stair', 'confidence': density}]
        return []

class CurbDetector:
    """Analyzes bottom 20% of frame for smaller drop signatures."""
    def analyze(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(0.8*h):h, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100) # Lower threshold than stairs
        
        density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        
        if density > config.CURB_SENSITIVITY:
            return [{'type': 'curb', 'confidence': density}]
        return []

class VisionAgent:
    """Orchestrator for all detectors."""
    def __init__(self):
        self.base_detector = MobileNetDetector()
        self.approach = ApproachDetector()
        self.static = StaticObstacleDetector()
        self.stair = StairDetector()
        self.curb = CurbDetector()

    def analyze(self, frame):
        # 1. Base detections
        raw_detections = self.base_detector.detect(frame)
        
        # 2. Run specialized detectors
        alerts = []
        alerts.extend(self.approach.analyze(raw_detections))
        alerts.extend(self.static.analyze(raw_detections))
        alerts.extend(self.stair.analyze(frame))
        alerts.extend(self.curb.analyze(frame))
        
        if not alerts:
            return None, raw_detections
            
        # 3. Sort by priority
        # PRIORITY = ["stair", "curb", "approach", "obstacle"]
        def priority_key(alert):
            try:
                return config.PRIORITY.index(alert['type'])
            except ValueError:
                return 99
        
        alerts.sort(key=priority_key)
        
        return alerts[0], raw_detections

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import config
from collections import deque

class SegFormerDetector:
    """Core Segmentation engine using SegFormer."""
    def __init__(self):
        print(f"Loading SegFormer ({config.SEGFORMER_MODEL})...")
        self.processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(config.SEGFORMER_MODEL)
        
        # Hardware Acceleration for Mac (MPS) or PC (CUDA)
        self.device = "cpu"
        if torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            
        print(f"Moving model to device: {self.device}")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        
        # Mapping common navigation classes based on ground truth labels
        # id2label is usually: {id: "vehicle-car", id: "human-person", etc.}
        self.nav_map = {
            'sidewalk': [k for k, v in self.id2label.items() if 'sidewalk' in v.lower()],
            'road':     [k for k, v in self.id2label.items() if 'road' in v.lower() and 'sidewalk' not in v.lower()],
            'curb':     [k for k, v in self.id2label.items() if 'curb' in v.lower()],
            'stair':    [k for k, v in self.id2label.items() if 'stair' in v.lower()],
            'pole':     [k for k, v in self.id2label.items() if 'pole' in v.lower()]
        }
        
        # Fixed palette: prioritize exact matches (BGR format for OpenCV)
        self.palette = np.random.randint(60, 230, (len(self.id2label), 3), dtype=np.uint8)
        class_colors = {
            'sidewalk': (232, 35, 244),  # Pink (BGR)
            'road':     (128, 64, 128),  # Purple (BGR)
            'curb':     (255, 165, 0),   # Orange (BGR)
            'pole':     (128, 128, 192), # Grey-Blue (BGR)
            'stair':    (200, 200, 0),   # Teal (BGR)
            'person':   (220, 20, 60),   # Red (BGR)
            'car':      (142, 0, 0),     # Blue (BGR)
            'wall':     (120, 120, 120), # Grey (BGR)
            'door':     (50, 100, 150),  # Brown (BGR)
            'void':     (200, 150, 200), # Light Violet (BGR)
        }
        for cls_id, label in self.id2label.items():
            l_lower = label.lower()
            for key, color in class_colors.items():
                if key == "car" and "caravan" in l_lower: continue
                if key in l_lower:
                    self.palette[int(cls_id)] = color
                    break

    def colorize_mask(self, mask):
        """Convert class ID mask to BGR color mask."""
        return self.palette[mask]

    def detect(self, frame):
        """Runs segmentation and returns mask + derived detections."""
        h, w = frame.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with torch.inference_mode():
            # Speed optimization: Process at smaller resolution
            inputs = self.processor(images=pil_img, 
                                    size={"height": config.DETECTION_SIZE, "width": config.DETECTION_SIZE},
                                    return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.device == "cuda" or self.device == "mps":
                with torch.autocast(device_type=self.device):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        # 1. Get low-res mask and confidence at native model scale
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf_mask_low, _ = probs.max(dim=1)
        mask_low = logits.argmax(dim=1)

        # 2. Only upsample the final result for the display output
        # Use 'nearest' for the integer mask to avoid color bleed between classes
        upsampled_mask = torch.nn.functional.interpolate(
            mask_low.unsqueeze(0).float(), size=(h, w), mode='nearest'
        ).squeeze().cpu().numpy().astype(np.uint8)
        
        # Keep internal masks for detection math
        conf_mask = torch.nn.functional.interpolate(
            conf_mask_low.unsqueeze(0), size=(h, w), mode='bilinear'
        ).squeeze().cpu().numpy()
        
        mask = upsampled_mask
        
        detections = []
        # Precise class mapping to avoid substring confusion (like "car" in "caravan")
        class_to_ids = {}
        for cls_id, label in self.id2label.items():
            l_lower = label.lower()
            if "person" in l_lower: class_to_ids.setdefault("person", []).append(cls_id)
            elif "car" in l_lower and "caravan" not in l_lower: class_to_ids.setdefault("car", []).append(cls_id)
            elif "bus" in l_lower: class_to_ids.setdefault("bus", []).append(cls_id)
            elif "truck" in l_lower: class_to_ids.setdefault("truck", []).append(cls_id)
            elif "motorcycle" in l_lower: class_to_ids.setdefault("motorcycle", []).append(cls_id)
            elif "bicycle" in l_lower: class_to_ids.setdefault("bicycle", []).append(cls_id)

        for cls_name, cls_ids in class_to_ids.items():
            cls_mask = np.isin(mask, cls_ids).astype(np.uint8) * 255
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w_b, h_b = cv2.boundingRect(cnt)
                area_ratio = (w_b * h_b) / (w * h)
                
                if area_ratio > 0.01:
                    # Calculate mean confidence for this detection
                    roi_conf = conf_mask[y:y+h_b, x:x+w_b]
                    roi_binary = (cls_mask[y:y+h_b, x:x+w_b] > 0)
                    if roi_binary.any():
                        conf = float(roi_conf[roi_binary].mean())
                    else:
                        conf = 0.0
                    
                    if conf > config.CONFIDENCE_THRESHOLD:
                        detections.append({
                            'class': cls_name,
                            'confidence': conf,
                            'bbox': (x, y, x + w_b, y + h_b),
                            'area_ratio': area_ratio,
                            'center_x': (x + w_b/2) / w
                        })
                    
        return mask, detections

    def analyze_hazards(self, mask):
        """Analyze bottom center zone for curbs/stairs. Uses EMA to only alert if hazard is GROWING/APPROACHING."""
        h, w = mask.shape
        hazard_h = int(h * config.HAZARD_ZONE_HEIGHT)
        
        margin = (1.0 - config.APPROACH_ZONE) / 2
        left_bound = int(w * margin)
        right_bound = int(w * (1 - margin))
        
        roi = mask[h - hazard_h:h, left_bound:right_bound]
        
        curb_pixels = np.isin(roi, self.nav_map['curb']).sum()
        stair_pixels = np.isin(roi, self.nav_map['stair']).sum()
        
        roi_size = roi.size
        stair_density = stair_pixels / roi_size
        curb_density = curb_pixels / roi_size
        
        min_density_threshold = (config.CURB_STAIR_MIN_PIXELS * config.APPROACH_ZONE) / roi_size
        
        alerts = []
        
        # Track history dynamically
        if not hasattr(self, 'hazard_history'):
            self.hazard_history = {'stair': deque(maxlen=5), 'curb': deque(maxlen=5)}
            self.hazard_emas = {'stair': 0.0, 'curb': 0.0}
            
        alpha = config.APPROACH_EMA_ALPHA
            
        for hazard_type, current_density in [('stair', stair_density), ('curb', curb_density)]:
            if current_density > min_density_threshold:
                # Update EMA
                prev_ema = self.hazard_emas[hazard_type]
                current_ema = (alpha * current_density) + ((1 - alpha) * prev_ema)
                self.hazard_emas[hazard_type] = current_ema
                
                hist = self.hazard_history[hazard_type]
                hist.append(current_ema)
                
                # We need a few frames to confirm it's actually approaching
                if len(hist) >= 3:
                    is_growing = hist[-1] > hist[0]
                    # Only alert if the density is getting larger (getting closer)
                    if is_growing:
                        alerts.append({'type': hazard_type, 'confidence': float(current_ema)})
            else:
                # Clear history if nothing is there to avoid stale data
                self.hazard_history[hazard_type].clear()
                self.hazard_emas[hazard_type] = 0.0
                
        return alerts

    def check_safe_path(self, mask):
        """Check if center path is mostly sidewalk/road."""
        h, w = mask.shape
        margin = (1.0 - config.APPROACH_ZONE) / 2
        path_mask = mask[:, int(w * margin):int(w * (1 - margin))]
        
        safe_ids = self.nav_map['sidewalk'] + self.nav_map['road']
        safe_pixels = np.isin(path_mask, safe_ids).sum()
        density = safe_pixels / path_mask.size
        
        return density > config.SAFE_PATH_MIN_DENSITY

class ApproachDetector:
    """Detects objects rapidly moving toward the user."""
    def __init__(self):
        self.history = {}
        self.emas = {}

    def analyze(self, detections):
        alerts = []
        current_classes = set()
        for det in detections:
            cls = det['class']
            if cls not in config.APPROACH_CLASSES: continue
            if det['area_ratio'] < config.APPROACH_MIN_SIZE: continue
            
            track_name = cls
            alpha = config.APPROACH_EMA_ALPHA
            prev_ema = self.emas.get(track_name, det['area_ratio'])
            current_ema = (alpha * det['area_ratio']) + ((1 - alpha) * prev_ema)
            self.emas[track_name] = current_ema
            
            if track_name not in self.history:
                self.history[track_name] = deque(maxlen=config.APPROACH_TREND_FRAMES)
            
            hist = self.history[track_name]
            hist.append(current_ema)
            current_classes.add(track_name)

            if len(hist) >= config.APPROACH_TREND_FRAMES:
                net_growth = (hist[-1] - hist[0]) / hist[0] if hist[0] > 0 else 0
                growing_steps = sum(1 for i in range(1, len(hist)) if hist[i] > hist[i-1])
                is_trending = (growing_steps >= (len(hist) * 0.8)) and (net_growth > 0)
                avg_growth = net_growth / (len(hist) - 1)
                
                if is_trending and avg_growth > config.APPROACH_SENSITIVITY:
                    alerts.append({'type': 'approach', 'class': cls, 'growth_rate': avg_growth, 'area_ratio': current_ema})

        for cls in list(self.history.keys()):
            if cls not in current_classes:
                self.history.pop(cls)
                self.emas.pop(cls, None)
        return alerts

class StaticObstacleDetector:
    """Detects obstacles in the way."""
    def __init__(self):
        self.counters = {}
        self.prev_areas = {}

    def analyze(self, detections):
        alerts = []
        current_classes = set()
        for det in detections:
            cls = det['class']
            if cls not in config.OBSTACLE_CLASSES: continue
            if det['area_ratio'] < config.OBSTACLE_MIN_SIZE: continue
            
            track_name = cls
            prev_area = self.prev_areas.get(track_name, 0)
            is_growing = det['area_ratio'] > prev_area
            
            if is_growing: self.counters[track_name] = self.counters.get(track_name, 0) + 1
            else: self.counters[track_name] = 0
            
            if self.counters[track_name] >= config.OBSTACLE_GROWTH_THRESHOLD:
                alerts.append({'type': 'obstacle', 'class': cls, 'area_ratio': det['area_ratio']})
            
            self.prev_areas[track_name] = det['area_ratio']
            current_classes.add(track_name)
            
        for cls in list(self.counters.keys()):
            if cls not in current_classes:
                self.counters.pop(cls)
                self.prev_areas.pop(cls, None)
        return alerts

class VisionAgent:
    """Orchestrator for SegFormer-based detection."""
    def __init__(self):
        self.seg_engine = SegFormerDetector()
        self.approach = ApproachDetector()
        self.static = StaticObstacleDetector()

    def analyze(self, frame):
        # 1. Run Segmentation
        mask, detections = self.seg_engine.detect(frame)
        
        # 2. Specialized Checks
        alerts = []
        alerts.extend(self.seg_engine.analyze_hazards(mask)) # Curbs / Stairs
        alerts.extend(self.approach.analyze(detections))    # Vehicles / People
        alerts.extend(self.static.analyze(detections))      # Static things
        
        # 3. Safe Path Status (Optional logging)
        is_safe = self.seg_engine.check_safe_path(mask)
        
        if not alerts:
            return None, (mask, detections)
            
        # 4. Priority Sorting
        def priority_key(alert):
            try: return config.PRIORITY.index(alert['type'])
            except ValueError: return 99
        
        alerts.sort(key=priority_key)
        
        return alerts[0], (mask, detections)

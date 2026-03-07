import cv2
import numpy as np
import config

try:
    from ultralytics import YOLO
except ImportError:
    # Fallback for non-Pi environments/during migration
    class YOLO:
        def __init__(self, model):
            print(f"Warning: ultralytics not found. Mocking YOLO with {model}.")
        def __call__(self, frame, verbose=False):
            class MockResult:
                def __init__(self):
                    self.boxes = type('obj', (object,), {
                        'cls': np.array([]),
                        'conf': np.array([]),
                        'xyxy': np.array([])
                    })
            return [MockResult()]

class ObjectDetector:
    """Handles object detection using Ultralytics YOLOv8."""
    
    def __init__(self):
        # Load the YOLOv8n model
        # On first run, this will download yolov8n.pt if not present
        self.model = YOLO(config.MODEL_WEIGHTS)
        
        # YOLOv8 COCO class names (standard)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect(self, frame):
        """
        Runs inference on the frame and returns sorted filtered detections.
        Each detection is a dict: {'class': str, 'confidence': float, 'box': [x1, y1, x2, y2], 'area_ratio': float}
        """
        h, w = frame.shape[:2]
        
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        # results is a list (usually len=1 for single frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                
                if confidence > config.CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls[0])
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        
                        # Check if it's a danger class
                        if class_name in config.DANGER_CLASSES:
                            # xyxy format: [x1, y1, x2, y2]
                            coords = box.xyxy[0].tolist()
                            left, top, right, bottom = map(int, coords)
                            
                            # Compute area ratio as distance proxy
                            box_area = (right - left) * (bottom - top)
                            frame_area = w * h
                            area_ratio = box_area / frame_area
                            
                            detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'box': [left, top, right, bottom],
                                'area_ratio': area_ratio
                            })
        
        # Sort by area_ratio (closest first)
        detections.sort(key=lambda x: x['area_ratio'], reverse=True)
        return detections

if __name__ == "__main__":
    # Test with a blank frame
    detector = ObjectDetector()
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    print(f"Detected {len(res)} objects in blank frame.")

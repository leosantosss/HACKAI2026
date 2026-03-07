# VisionAid Configuration Constants

# Hardware Pins
BUZZER_GPIO_PIN = 18

# Buzzer Settings
BUZZER_FREQUENCY = 2000

# AI Detection Settings
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_TIME = 1.5  # Seconds between notifications for the same class

# Model Files
MODEL_WEIGHTS = "yolov8n.pt"  # Will be downloaded automatically by ultralytics

# Distance Proxy (Area Ratio)
# ratio = (box_width * box_height) / (frame_width * frame_height)
PROXIMITY_FAR = 0.05
PROXIMITY_MEDIUM = 0.15
PROXIMITY_CLOSE = 0.30

# Danger Classes (COCO dataset indices or names)
# Indices for MobileNet SSD v2 COCO:
# 1: person, 2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck, 10: traffic light, 11: fire hydrant, 13: stop sign, 14: parking meter, 15: bench
DANGER_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", 
    "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
]

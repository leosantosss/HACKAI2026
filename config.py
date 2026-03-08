# VisionAid Configuration Constants

# Hardware
BUZZER_GPIO_PIN = 18
BUZZER_FREQUENCY = 2000

# Model Files
MODEL_PROTO = "ssd_mobilenet_v2_coco.pbtxt"
MODEL_WEIGHTS = "frozen_inference_graph.pb"
CONFIDENCE_THRESHOLD = 0.5
SAFETY_CONFIDENCE_THRESHOLD = 0.25 # Lower threshold for objects in the center "path"

# Detection zones
APPROACH_ZONE = 0.5              # center fraction of frame that counts as "in path"

# Sensitivities
APPROACH_SENSITIVITY = 0.12      # Slightly more sensitive but smoothed
APPROACH_TREND_FRAMES = 6        # Increased to 6 for even more stability
APPROACH_EMA_ALPHA = 0.3         # Smoothing factor (0.0 to 1.0, lower is smoother)
APPROACH_MIN_SIZE = 0.10         # Catch threats when they are roughly 10% of frame
OBSTACLE_GROWTH_THRESHOLD = 5    # consecutive frames of growth
OBSTACLE_MIN_SIZE = 0.10          # minimum bbox area ratio to consider an obstacle
STAIR_CONFIDENCE_THRESHOLD = 0.6 # edge density (fraction) to trigger stair alert
CURB_SENSITIVITY = 0.4           # minimum drop size to detect a curb (edge density)

# Cooldowns (seconds)
COOLDOWN_APPROACH = .0
COOLDOWN_OBSTACLE = 2.0
COOLDOWN_STAIR = 3.0
COOLDOWN_CURB = 2.0

# Behavior
ALERT_SPEED_SCALING = True       # pulse rate increases as object gets closer
SHOW_DISPLAY = True              # debug display
INCLUDE_UNKNOWN_OBJECTS = True   # Alert on any object in path, even if class is unrecognized

# Classes watched by approach and obstacle detectors
APPROACH_CLASSES = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
OBSTACLE_CLASSES = ["person", "car", "bus", "truck", "motorcycle", "bicycle", 
                    "bench", "fire hydrant", "traffic light", "stop sign"]

# Priority order
PRIORITY = ["stair", "curb", "approach", "obstacle"]

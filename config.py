# Vizzion Configuration Constants

# Hardware
BUZZER_GPIO_PIN = 18
BUZZER_FREQUENCY = 2000

# Model Files
SEGFORMER_MODEL = "vizzion-navigation-master"
CONFIDENCE_THRESHOLD = 0.5
SAFETY_CONFIDENCE_THRESHOLD = 0.25 

# Segmentation Thresholds
SAFE_PATH_MIN_DENSITY = 0.6      # % of center path that must be "sidewalk/road"
CURB_STAIR_MIN_PIXELS = 1000     # Minimum pixel count in hazard zone to trigger alert
HAZARD_ZONE_HEIGHT = 0.25        # Bottom % of frame to analyze for curbs/stairs

# Detection zones
APPROACH_ZONE = 0.5              # center fraction of frame that counts as "in path"

# Sensitivities
APPROACH_SENSITIVITY = 0.12      
APPROACH_TREND_FRAMES = 6        
APPROACH_EMA_ALPHA = 0.3         
APPROACH_MIN_SIZE = 0.10         
OBSTACLE_GROWTH_THRESHOLD = 5    
OBSTACLE_MIN_SIZE = 0.10          

# Cooldowns (seconds)
COOLDOWN_APPROACH = 1.0
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

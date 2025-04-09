import os

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Video settings
VIDEO_SOURCE = 0  # 0 for default camera
WIDTH = 640  # Reduced for better performance
HEIGHT = 480

# Detection intervals (in seconds)
MOTION_DETECTION_INTERVAL = 0.1
FACE_RECOGNITION_INTERVAL = 1.0
ALERT_SUMMARY_INTERVAL = 60.0

# Alert settings
ALERT_LOG_DIR = os.path.join(PROJECT_ROOT, 'data', 'alerts')
ALERT_LOG_FILE = os.path.join(ALERT_LOG_DIR, 'alerts.json')

# Ensure alert directory exists
os.makedirs(ALERT_LOG_DIR, exist_ok=True)

# Time windows for alert summaries (in seconds)
TIME_WINDOWS = {
    "short": 300,    # 5 minutes
    "medium": 3600,  # 1 hour
    "long": 86400    # 24 hours
}

# Object categories and their settings
OBJECT_CATEGORIES = {
    "HIGH_PRIORITY": {
        "objects": ["person", "gun", "knife", "phone"],
        "threshold": 0.5,
        "alert_cooldown": 5.0
    },
    "MEDIUM_PRIORITY": {
        "objects": ["backpack", "handbag", "laptop"],
        "threshold": 0.4,
        "alert_cooldown": 10.0
    },
    "LOW_PRIORITY": {
        "objects": ["chair", "book", "bottle"],
        "threshold": 0.3,
        "alert_cooldown": 15.0
    },
    "IGNORE": {
        "objects": ["tv", "remote", "keyboard"],
        "threshold": 0.0,
        "alert_cooldown": 0.0
    }
}

# Minimum confidence threshold for detection
MIN_CONFIDENCE = 0.3

# Model settings
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolov8n.pt')

# Motion Detection Settings
MOTION_THRESHOLD = 5000

# Face Recognition Settings
KNOWN_FACES_DIR = os.path.join(PROJECT_ROOT, 'known_faces')
FACE_MATCH_THRESHOLD = 0.65

# Alert Settings
ALERT_BACKUP_INTERVAL = 60  # Backup alerts.json every N seconds

# Model Settings
MODEL_PATH = "yolo-Weights/yolov8n.pt"
MIN_CONFIDENCE = 0.4

# Motion Detection Settings
MOTION_THRESHOLD = 5000
MOTION_DETECTION_INTERVAL = 2  # Process every Nth frame for motion

# Face Recognition Settings
KNOWN_FACES_DIR = "known_faces"
FACE_RECOGNITION_INTERVAL = 15  # Process every Nth frame for face recognition
FACE_MATCH_THRESHOLD = 0.65

# Alert Settings
ALERT_LOG_FILE = "alerts.json"
ALERT_SUMMARY_INTERVAL = 300  # Generate summary every N seconds
ALERT_BACKUP_INTERVAL = 60  # Backup alerts.json every N seconds

# Object Categories and their confidence thresholds
OBJECT_CATEGORIES = {
    "HIGH_PRIORITY": {
        "objects": ["person", "car", "truck", "bus"],
        "threshold": 0.4,
        "alert_cooldown": 30,  # Seconds before alerting for same object again
    },
    "MEDIUM_PRIORITY": {
        "objects": ["bicycle", "motorbike", "dog", "cat", "cell phone", "laptop"],
        "threshold": 0.5,
        "alert_cooldown": 60,
    },
    "LOW_PRIORITY": {
        "objects": ["bird", "chair", "bottle"],
        "threshold": 0.6,
        "alert_cooldown": 120,
    },
    "IGNORE": {
        "objects": ["tree", "leaves", "clouds"],
        "threshold": 1.0,  # Will never trigger
        "alert_cooldown": 0,
    }
}

# Time windows for alert analysis (in seconds)
TIME_WINDOWS = {
    "short": 300,    # 5 minutes
    "medium": 1800,  # 30 minutes
    "long": 3600     # 1 hour
} 
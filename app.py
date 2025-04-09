from ultralytics import YOLO
import cv2
import math
import numpy as np
import os
from datetime import datetime
import json
import time
import pandas as pd
from collections import Counter

# Define constants
WIDTH, HEIGHT = 640, 480
MODEL_PATH = "yolo-Weights/yolov8n.pt"
MOTION_THRESHOLD = 5000  # Adjust based on sensitivity
MIN_CONFIDENCE = 0.3  # Lowered confidence threshold to detect more objects
KNOWN_FACES_DIR = "known_faces"
ALERT_LOG_FILE = "alerts.json"
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file for testing
#VIDEO_SOURCE = "test_video.mp4"  # Replace with your video filename

# YOLO class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Define alert categories
ALERT_CATEGORIES = {
    "HIGH": ["person", "car", "truck", "bus"],
    "MEDIUM": ["bicycle", "motorbike", "dog", "cat"],
    "LOW": ["bird", "chair", "bottle"],
    "IGNORE": ["tree", "leaves", "clouds"]
}

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face recognition parameters
FACE_MATCH_THRESHOLD = 0.6  # Lower values make matching more strict
FACE_RECOGNITION_INTERVAL = 30  # Only perform face recognition every N frames
frame_counter = 0
last_known_face_result = False

class KnownFace:
    def __init__(self, name, face_encoding, face_rect):
        self.name = name
        self.face_encoding = face_encoding
        self.face_rect = face_rect

def initialize_alerts_file():
    """Initialize the alerts.json file with an empty array if it doesn't exist"""
    if not os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, 'w') as f:
            json.dump([], f, indent=4)
        print(f"Created new {ALERT_LOG_FILE} file")
    else:
        print(f"{ALERT_LOG_FILE} already exists")

def get_alerts_by_date(start_date=None, end_date=None, status=None):
    """Get alerts filtered by date range and status"""
    try:
        with open(ALERT_LOG_FILE, 'r') as f:
            alerts = json.load(f)
        
        if start_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            alerts = [a for a in alerts if datetime.strptime(a['timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date]
        
        if end_date:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            alerts = [a for a in alerts if datetime.strptime(a['timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date]
        
        if status:
            alerts = [a for a in alerts if a['status'] == status]
        
        return alerts
    except Exception as e:
        print(f"Error getting alerts: {str(e)}")
        return []

def update_alert_status(alert_index, new_status):
    """Update the status of a specific alert"""
    try:
        with open(ALERT_LOG_FILE, 'r') as f:
            alerts = json.load(f)
        
        if 0 <= alert_index < len(alerts):
            alerts[alert_index]['status'] = new_status
            with open(ALERT_LOG_FILE, 'w') as f:
                json.dump(alerts, f, indent=4)
            return True
        return False
    except Exception as e:
        print(f"Error updating alert status: {str(e)}")
        return False

def get_alert_statistics():
    """Get statistics about alerts"""
    try:
        with open(ALERT_LOG_FILE, 'r') as f:
            alerts = json.load(f)
        
        if not alerts:
            return {
                "total_alerts": 0,
                "by_type": {},
                "by_status": {},
                "average_confidence": 0
            }
        
        # Count alerts by type
        type_counts = Counter(a['object'] for a in alerts)
        
        # Count alerts by status
        status_counts = Counter(a['status'] for a in alerts)
        
        # Calculate average confidence
        avg_confidence = sum(a['confidence'] for a in alerts) / len(alerts)
        
        return {
            "total_alerts": len(alerts),
            "by_type": dict(type_counts),
            "by_status": dict(status_counts),
            "average_confidence": avg_confidence
        }
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
        return None

def export_alerts_to_csv(filename="alerts_export.csv"):
    """Export alerts to a CSV file"""
    try:
        with open(ALERT_LOG_FILE, 'r') as f:
            alerts = json.load(f)
        
        if alerts:
            df = pd.DataFrame(alerts)
            df.to_csv(filename, index=False)
            print(f"Alerts exported to {filename}")
            return True
        return False
    except Exception as e:
        print(f"Error exporting alerts: {str(e)}")
        return False

def get_face_encoding(face_img):
    """Convert face image to a feature vector using HOG and LBP"""
    # Resize image to a standard size
    face_img = cv2.resize(face_img, (64, 64))
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Calculate HOG features
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(gray)
    
    # Calculate LBP features
    lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Combine HOG and LBP features
    combined_features = np.concatenate([hog_features.flatten(), lbp.flatten()])
    
    # Normalize the feature vector
    if np.any(combined_features):
        combined_features = combined_features / np.linalg.norm(combined_features)
    
    return combined_features

def compare_faces(known_encoding, face_encoding):
    """Compare two face encodings using cosine similarity"""
    if len(known_encoding) != len(face_encoding):
        return 0
    
    similarity = np.dot(known_encoding, face_encoding)
    return similarity

def load_known_faces():
    """Load known faces and their encodings"""
    known_faces = []
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        return known_faces
    
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            try:
                # Load image
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load image: {filename}")
                    continue
                
                # Detect face
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_img = image[y:y+h, x:x+w]
                    face_encoding = get_face_encoding(face_img)
                    name = os.path.splitext(filename)[0]
                    known_faces.append(KnownFace(name, face_encoding, faces[0]))
                    print(f"Loaded known face: {name}")
                else:
                    print(f"No face detected in {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"Loaded {len(known_faces)} known faces")
    return known_faces

def initialize_camera(width=WIDTH, height=HEIGHT):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def detect_motion(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > MOTION_THRESHOLD:
            motion_detected = True
            break
    
    return motion_detected

def check_face_recognition(frame, known_faces):
    """Check if any faces in the frame match known faces"""
    global frame_counter, last_known_face_result
    
    # Only perform face recognition every N frames
    frame_counter += 1
    if frame_counter % FACE_RECOGNITION_INTERVAL != 0:
        return last_known_face_result
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # For each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_encoding = get_face_encoding(face_img)
        
        # Compare with known faces
        for known_face in known_faces:
            similarity = compare_faces(known_face.face_encoding, face_encoding)
            if similarity > FACE_MATCH_THRESHOLD:
                # Draw green rectangle around recognized face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, known_face.name, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                last_known_face_result = True
                return True
        
        # Draw red rectangle around unknown face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Unknown", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    last_known_face_result = False
    return False

def log_alert(alert_type, object_name, confidence, timestamp):
    """Log an alert to the alerts.json file"""
    try:
        # Read existing alerts
        if os.path.exists(ALERT_LOG_FILE):
            with open(ALERT_LOG_FILE, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []
        
        # Create new alert
        alert = {
            "type": alert_type,
            "object": object_name,
            "confidence": float(confidence),
            "timestamp": timestamp,
            "status": "new"
        }
        
        # Add alert to list
        alerts.append(alert)
        
        # Write back to file
        with open(ALERT_LOG_FILE, 'w') as f:
            json.dump(alerts, f, indent=4)
            
        print(f"Alert logged: {object_name} at {timestamp}")
        
    except Exception as e:
        print(f"Error logging alert: {str(e)}")

def should_alert(object_name, confidence):
    for category, objects in ALERT_CATEGORIES.items():
        if object_name in objects:
            if category == "HIGH":
                return True
            elif category == "MEDIUM" and confidence > 0.5:
                return True
            elif category == "LOW" and confidence > 0.7:
                return True
            elif category == "IGNORE":
                return False
    return False

def draw_bounding_box(img, box, cls, confidence, object_name):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color = (0, 255, 0) if object_name in ALERT_CATEGORIES["HIGH"] else (0, 165, 255)
    label = f"{object_name} {confidence:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_frame(frame, model, known_faces):
    """Process a single frame for object detection and alerting"""
    # Check for known faces first
    is_known_face = check_face_recognition(frame, known_faces)
    
    # Process objects
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]
            cls = int(box.cls[0])
            object_name = CLASS_NAMES[cls]
            
            # Only alert for person if face is not recognized
            if object_name == "person" and is_known_face:
                continue
                
            if confidence > MIN_CONFIDENCE and should_alert(object_name, confidence):
                draw_bounding_box(frame, box, cls, confidence, object_name)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_alert("object_detection", object_name, float(confidence), timestamp)
    
    return frame

def main():
    # Initialize alerts file
    initialize_alerts_file()
    
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    
    # Initialize video capture
    if isinstance(VIDEO_SOURCE, int):
        cap = cv2.VideoCapture(VIDEO_SOURCE)  # Webcam
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)  # Video file
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # Load known faces
    known_faces = load_known_faces()
    
    # Initialize motion detection
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
            
        # Check for motion
        motion_detected = detect_motion(frame1, frame2)
        
        if motion_detected:
            # Process frame for object detection
            frame = process_frame(frame, model, known_faces)
        
        # Update frames for motion detection
        frame1 = frame2
        frame2 = frame
        
        # Display the video feed with detections
        cv2.imshow('Enhanced Security Camera', frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Press 's' to show statistics
            stats = get_alert_statistics()
            print("\nAlert Statistics:")
            print(f"Total Alerts: {stats['total_alerts']}")
            print(f"Average Confidence: {stats['average_confidence']:.2f}")
            print("\nAlerts by Type:")
            for obj, count in stats['by_type'].items():
                print(f"{obj}: {count}")
            print("\nAlerts by Status:")
            for status, count in stats['by_status'].items():
                print(f"{status}: {count}")
        elif key == ord('e'):  # Press 'e' to export alerts
            export_alerts_to_csv()
        elif key == ord('p'):  # Press 'p' to print current alerts
            with open(ALERT_LOG_FILE, 'r') as f:
                alerts = json.load(f)
                print("\nCurrent Alerts:")
                for alert in alerts:
                    print(f"Object: {alert['object']}, Confidence: {alert['confidence']}, Time: {alert['timestamp']}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

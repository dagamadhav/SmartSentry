from ultralytics import YOLO
import cv2
import math
import numpy as np
import os
from datetime import datetime
import json
import time
import pandas as pd
from collections import Counter, defaultdict
from src.alert_manager import AlertManager
from src.notification_manager import NotificationManager

# Define constants
WIDTH, HEIGHT = 640, 480
MODEL_PATH = "yolo-Weights/yolov8n.pt"
MOTION_THRESHOLD = 5000  # Adjust based on sensitivity
MIN_CONFIDENCE = 0.3  # Lowered confidence threshold to detect more objects
KNOWN_FACES_DIR = "known_faces"
VIDEO_SOURCE = "test1.mp4"  # Path to your test video file
ALERT_COOLDOWN = 120  # 2 minutes cooldown between alerts for the same object

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
    "HIGH_PRIORITY": ["person", "car", "truck", "bus"],
    "MEDIUM_PRIORITY": ["bicycle", "motorbike", "dog", "cat"],
    "LOW_PRIORITY": ["bird", "chair", "bottle"],
    "IGNORED": ["tree", "leaves", "clouds"]
}

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face recognition parameters
FACE_MATCH_THRESHOLD = 0.6  # Lower values make matching more strict
FACE_RECOGNITION_INTERVAL = 30  # Only perform face recognition every N frames
frame_counter = 0
last_known_face_result = False
last_known_face_name = None

class KnownFace:
    def __init__(self, name, face_encoding, face_rect):
        self.name = name
        self.face_encoding = face_encoding
        self.face_rect = face_rect

class AlertTracker:
    def __init__(self, cooldown_period=ALERT_COOLDOWN):
        self.last_alert_times = defaultdict(float)
        self.cooldown_period = cooldown_period
        self.known_person_detected = False
        self.known_person_name = None
        self.last_known_person_time = 0
    
    def can_alert(self, object_name, confidence, is_known_person=False, person_name=None):
        current_time = time.time()
        
        # Handle known person detection
        if is_known_person and person_name:
            if current_time - self.last_known_person_time >= self.cooldown_period:
                self.known_person_detected = True
                self.known_person_name = person_name
                self.last_known_person_time = current_time
                return True
            return False
        
        # Handle regular object detection
        if current_time - self.last_alert_times[object_name] >= self.cooldown_period:
            self.last_alert_times[object_name] = current_time
            return True
        return False
    
    def get_known_person_info(self):
        return self.known_person_detected, self.known_person_name

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
    cap = cv2.VideoCapture(VIDEO_SOURCE)
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
    global frame_counter, last_known_face_result, last_known_face_name
    
    # Only perform face recognition every N frames
    frame_counter += 1
    if frame_counter % FACE_RECOGNITION_INTERVAL != 0:
        return last_known_face_result, last_known_face_name
    
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
                # Add name label with background for better visibility
                text = f"{known_face.name}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y-30), (x + text_size[0], y), (0, 255, 0), -1)
                cv2.putText(frame, text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                last_known_face_result = True
                last_known_face_name = known_face.name
                return True, known_face.name
        
        # Draw red rectangle around unknown face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Add "Unknown" label with background
        text = "Unknown"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-30), (x + text_size[0], y), (0, 0, 255), -1)
        cv2.putText(frame, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    last_known_face_result = False
    last_known_face_name = None
    return False, None

def get_alert_category(object_name, confidence):
    """Determine alert category based on object and confidence"""
    for category, objects in ALERT_CATEGORIES.items():
        if object_name in objects:
            if category == "HIGH_PRIORITY":
                return category
            elif category == "MEDIUM_PRIORITY" and confidence > 0.5:
                return category
            elif category == "LOW_PRIORITY" and confidence > 0.7:
                return category
            elif category == "IGNORED":
                return "LOW_PRIORITY"  # Convert IGNORED to LOW_PRIORITY for logging
    return "LOW_PRIORITY"  # Default category

def draw_bounding_box(img, box, cls, confidence, object_name, is_known_person=False, person_name=None):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    if is_known_person and person_name:
        # Known person
        color = (0, 255, 0)  # Green
        label = f"{person_name}"
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Add name label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x1, y1-30), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        # Unknown person or other object
        color = (0, 165, 255) if object_name in ALERT_CATEGORIES["HIGH_PRIORITY"] else (0, 165, 255)
        label = f"{object_name} {confidence:.2f}"
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Add label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x1, y1-30), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def create_ui_overlay(frame, alert_summary, performance_metrics):
    """Create a clean, modern UI overlay"""
    # Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add alert summary
    if alert_summary:
        alert_text = f"Alerts: {alert_summary['alert_summary']['total_alerts']}"
        cv2.putText(frame, alert_text, (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add performance metrics
    if performance_metrics:
        fps_text = f"FPS: {performance_metrics['fps']:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def process_frame(frame, model, known_faces, alert_manager, alert_tracker, notification_manager):
    """Process a single frame for object detection and alerting"""
    # Check for known faces first
    is_known_face, person_name = check_face_recognition(frame, known_faces)
    
    # Process objects
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            object_name = CLASS_NAMES[cls]
            
            # Only process detections with confidence > 0.5
            if confidence > 0.5:
                # Handle person detection
                if object_name == "person":
                    if is_known_face:
                        # Known person detected
                        if alert_tracker.can_alert(object_name, confidence, True, person_name):
                            category = "HIGH_PRIORITY"
                            draw_bounding_box(frame, box, cls, confidence, object_name, True, person_name)
                            alert_id = f"known_{person_name}_{time.time()}"
                            alert_manager.add_alert(
                                f"Known Person: {person_name}",
                                confidence,
                                category,
                                f"Detected known person: {person_name}"
                            )
                            notification_manager.add_alert(alert_id, category, f"Known person detected: {person_name}")
                    else:
                        # Unknown person detected
                        if alert_tracker.can_alert(object_name, confidence):
                            category = get_alert_category(object_name, confidence)
                            draw_bounding_box(frame, box, cls, confidence, object_name)
                            alert_id = f"unknown_{time.time()}"
                            alert_manager.add_alert(
                                object_name,
                                confidence,
                                category,
                                f"Detected unknown person with {confidence:.2f} confidence"
                            )
                            if category == "HIGH_PRIORITY":
                                notification_manager.add_alert(alert_id, category, "Unknown person detected")
                else:
                    # Handle other objects
                    if alert_tracker.can_alert(object_name, confidence):
                        category = get_alert_category(object_name, confidence)
                        if category != "IGNORED":
                            draw_bounding_box(frame, box, cls, confidence, object_name)
                            alert_id = f"{object_name}_{time.time()}"
                            alert_manager.add_alert(
                                object_name,
                                confidence,
                                category,
                                f"Detected {object_name} with {confidence:.2f} confidence"
                            )
                            if category == "HIGH_PRIORITY":
                                notification_manager.add_alert(alert_id, category, f"High priority object detected: {object_name}")
            else:
                # Still draw bounding boxes for low confidence detections but don't save alerts
                draw_bounding_box(frame, box, cls, confidence, object_name)
    
    return frame

def main():
    # Initialize managers
    alert_manager = AlertManager()
    alert_manager.start_processing()
    alert_tracker = AlertTracker()
    notification_manager = NotificationManager()
    notification_manager.start()
    
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
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
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # Check for motion
            motion_detected = detect_motion(frame1, frame2)
            
            if motion_detected:
                # Process frame
                frame = process_frame(frame, model, known_faces, alert_manager, alert_tracker, notification_manager)
            
            # Update frames for motion detection
            frame1 = frame2
            frame2 = frame
            
            # Get current metrics
            alert_summary = alert_manager.get_alert_summary()
            performance_metrics = {'fps': fps}
            
            # Add UI overlay
            frame = create_ui_overlay(frame, alert_summary, performance_metrics)
            
            # Display the video feed
            cv2.imshow('Enhanced Security Camera', frame)
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if alert_summary:
                    print("\nAlert Summary:")
                    print(f"Total Alerts: {alert_summary['alert_summary']['total_alerts']}")
                    print(f"System Accuracy: {alert_summary['system_performance']['accuracy']}")
                    print("\nAlerts by Category:")
                    for category, count in alert_summary['alert_summary']['categories'].items():
                        print(f"{category}: {count}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_manager.stop_processing()
        notification_manager.stop()

if __name__ == "__main__":
    main()

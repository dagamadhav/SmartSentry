import cv2
import torch
from ultralytics import YOLO
import numpy as np
from ..config.settings import MODEL_PATH, MIN_CONFIDENCE, OBJECT_CATEGORIES

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model and set up detection parameters"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Set device and memory settings
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            
            # Load model with error handling
            self.model = YOLO(MODEL_PATH)
            self.model.to(self.device)
            self.class_names = self.model.names
            
            # Warm up the model
            dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
            _ = self.model(dummy_input)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
        
        # Define object categories with more specific classifications
        self.object_categories = {
            "HUMAN": {
                "objects": ["person"],
                "threshold": 0.5
            },
            "ANIMAL": {
                "objects": ["cat", "dog", "bird", "horse", "sheep", "cow", "bear", "zebra", "giraffe"],
                "threshold": 0.4
            },
            "VEHICLE": {
                "objects": ["car", "truck", "bus", "motorcycle", "bicycle"],
                "threshold": 0.4
            },
            "HIGH_PRIORITY": {
                "objects": ["gun", "knife", "phone", "laptop", "backpack"],
                "threshold": 0.5
            },
            "LOW_PRIORITY": {
                "objects": ["chair", "book", "bottle", "tv", "remote"],
                "threshold": 0.3
            }
        }

    def get_category(self, object_name):
        """Get the category for a detected object"""
        for category, info in self.object_categories.items():
            if object_name in info["objects"]:
                return category
        return "UNKNOWN"

    def should_detect(self, object_name, confidence):
        """Determine if an object should be detected based on confidence threshold"""
        for category, info in self.object_categories.items():
            if object_name in info["objects"]:
                return confidence >= info["threshold"]
        return confidence >= MIN_CONFIDENCE

    def detect_objects(self, frame):
        """Detect objects in frame and return results"""
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection with error handling
            with torch.cuda.amp.autocast() if self.device == 'cuda' else nullcontext():
                results = self.model(frame_rgb, stream=True, device=self.device)
                detections = []
                
                for r in results:
                    for box in r.boxes:
                        try:
                            confidence = float(box.conf[0])
                            cls = int(box.cls[0])
                            object_name = self.class_names[cls]
                            
                            if self.should_detect(object_name, confidence):
                                category = self.get_category(object_name)
                                detection = {
                                    "object": object_name,
                                    "confidence": confidence,
                                    "category": category,
                                    "box": box.xyxy[0].cpu().numpy()
                                }
                                detections.append(detection)
                        except Exception as e:
                            print(f"Error processing detection: {str(e)}")
                            continue
                
                return detections
                
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return []

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detected objects"""
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["box"])
            object_name = detection["object"]
            confidence = detection["confidence"]
            category = detection["category"]
            
            # Color based on category
            colors = {
                "HUMAN": (0, 255, 0),        # Green
                "ANIMAL": (255, 165, 0),     # Orange
                "VEHICLE": (255, 255, 0),    # Yellow
                "HIGH_PRIORITY": (0, 0, 255), # Red
                "LOW_PRIORITY": (128, 128, 128), # Gray
                "UNKNOWN": (255, 0, 255)     # Magenta
            }
            color = colors.get(category, (255, 0, 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{object_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw category
            cv2.putText(frame, category, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

class nullcontext:
    """Context manager that does nothing"""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass 
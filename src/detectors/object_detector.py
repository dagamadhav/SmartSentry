from ultralytics import YOLO
import cv2
import numpy as np
from ..config.settings import MODEL_PATH, OBJECT_CATEGORIES, MIN_CONFIDENCE

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names
        
    def get_category(self, object_name):
        """Determine the category of an object"""
        for category, info in OBJECT_CATEGORIES.items():
            if object_name in info["objects"]:
                return category
        return None
        
    def should_detect(self, object_name, confidence):
        """Check if object should be detected based on category thresholds"""
        category = self.get_category(object_name)
        if category:
            return confidence >= OBJECT_CATEGORIES[category]["threshold"]
        return confidence >= MIN_CONFIDENCE
    
    def detect_objects(self, frame):
        """Detect objects in frame and return results"""
        try:
            results = self.model(frame, stream=True)
            detections = []
            
            for r in results:
                for box in r.boxes:
                    try:
                        confidence = float(box.conf[0])
                        cls = int(box.cls[0])
                        object_name = self.class_names[cls]
                        
                        if self.should_detect(object_name, confidence):
                            category = self.get_category(object_name) or "OTHER"
                            detection = {
                                "object": object_name,
                                "confidence": confidence,
                                "category": category,
                                "box": box.xyxy[0].cpu().numpy()  # Convert to numpy array
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
                "HIGH_PRIORITY": (0, 0, 255),     # Red
                "MEDIUM_PRIORITY": (0, 165, 255),  # Orange
                "LOW_PRIORITY": (0, 255, 0),       # Green
                "IGNORE": (128, 128, 128),         # Gray
                "OTHER": (255, 0, 255)             # Magenta
            }
            color = colors.get(category, (255, 0, 255))  # Default to magenta if category not found
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{object_name} ({confidence:.2f})"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - label_height - 5), 
                (x1 + label_width + 5, y1), 
                color, 
                -1
            )
            cv2.putText(
                frame, 
                label, 
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        return frame 
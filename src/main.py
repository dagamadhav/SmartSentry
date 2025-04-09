import cv2
import time
import numpy as np
from datetime import datetime
import threading
import gc

from src.config.settings import (
    WIDTH, HEIGHT, VIDEO_SOURCE, MOTION_DETECTION_INTERVAL,
    FACE_RECOGNITION_INTERVAL, ALERT_SUMMARY_INTERVAL
)
from src.detectors.object_detector import ObjectDetector
from src.detectors.face_detector import FaceDetector
from src.alerts.alert_manager import AlertManager

class SecurityCamera:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.alert_manager = AlertManager()
        
        self.frame_count = 0
        self.last_summary_time = time.time()
        self.running = False
        self.paused = False
        self.last_gc_time = time.time()
        self.gc_interval = 60  # Run garbage collection every 60 seconds
        
    def initialize_video(self):
        """Initialize video capture"""
        try:
            # Try different camera indices if needed
            for i in range(3):  # Try first 3 camera indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Successfully opened camera {i}")
                    break
                cap.release()
            
            if not cap.isOpened():
                raise ValueError("Could not open any video source")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            
            # Test if we can actually read a frame
            ret, _ = cap.read()
            if not ret:
                raise ValueError("Could not read from camera")
            
            return cap
        except Exception as e:
            print(f"Error initializing video: {str(e)}")
            raise
    
    def process_frame(self, frame):
        """Process a single frame"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (640, 480))
            
            # Detect faces first
            small_frame, recognized_faces = self.face_detector.detect_faces(small_frame)
            known_faces = [face for face in recognized_faces if face['name'] != "Unknown"]
            
            # Detect objects
            detections = self.object_detector.detect_objects(small_frame)
            
            # Filter out person detections if face is recognized
            if known_faces:
                detections = [d for d in detections if d['object'] != 'person']
            
            # Scale detections back to original frame size
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            
            for detection in detections:
                detection['box'] = [
                    detection['box'][0] * scale_x,
                    detection['box'][1] * scale_y,
                    detection['box'][2] * scale_x,
                    detection['box'][3] * scale_y
                ]
            
            # Draw detections and generate alerts
            frame = self.object_detector.draw_detections(frame, detections)
            
            # Generate alerts
            for detection in detections:
                self.alert_manager.add_alert(
                    detection['object'],
                    detection['confidence'],
                    detection['category']
                )
            
            return frame
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame
    
    def show_summary(self):
        """Display alert summary"""
        summary = self.alert_manager.get_alert_summary()
        print("\n=== Alert Summary ===")
        print(f"Time Window: {summary['window_duration']}")
        print(f"Total Alerts: {summary['total_alerts']}")
        print("\nTop Objects Detected:")
        for obj, count in summary['top_objects'].items():
            duration = self.alert_manager.get_object_duration(obj)
            print(f"  {obj}: {count} alerts ({duration/60:.1f} minutes)")
        print("\nBy Category:")
        for category, count in summary['by_category'].items():
            print(f"  {category}: {count}")
        print(f"Average Confidence: {summary['average_confidence']:.2f}")
        print("==================\n")
    
    def run(self):
        """Main loop"""
        try:
            cap = self.initialize_video()
            self.running = True
            
            while self.running:
                if not self.paused:
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to grab frame, attempting to reinitialize camera...")
                            cap.release()
                            cap = self.initialize_video()
                            continue
                        
                        self.frame_count += 1
                        
                        # Process frame
                        processed_frame = self.process_frame(frame)
                        
                        # Show frame
                        cv2.imshow('Security Camera', processed_frame)
                        
                        # Check for periodic summary
                        current_time = time.time()
                        if current_time - self.last_summary_time >= ALERT_SUMMARY_INTERVAL:
                            self.show_summary()
                            self.last_summary_time = current_time
                        
                        # Run garbage collection periodically
                        if current_time - self.last_gc_time >= self.gc_interval:
                            gc.collect()
                            self.last_gc_time = current_time
                    
                    except Exception as e:
                        print(f"Error in frame processing: {str(e)}")
                        continue
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                elif key == ord('s'):
                    self.show_summary()
                elif key == ord('e'):
                    self.alert_manager.export_alerts()
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            raise
        
        finally:
            self.running = False
            gc.collect()  # Clean up before exit

def main():
    try:
        camera = SecurityCamera()
        camera.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 
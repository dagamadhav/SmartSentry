import cv2
import time
import numpy as np
from datetime import datetime
import threading
import gc
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from src.config.settings import (
    WIDTH, HEIGHT, VIDEO_SOURCE, MOTION_DETECTION_INTERVAL,
    FACE_RECOGNITION_INTERVAL, ALERT_SUMMARY_INTERVAL
)
from src.detectors.object_detector import ObjectDetector
from src.detectors.face_detector import FaceDetector
from src.alert_manager import AlertManager

class LLaVAAnalyzer:
    def __init__(self):
        """Initialize LLaVA model for scene analysis"""
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
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Warm up the model
            if self.device == 'cuda':
                dummy_input = torch.zeros((1, 3, 224, 224), device=self.device)
                _ = self.model(dummy_input)
            
            self.last_analysis = time.time()
            self.analysis_interval = 30  # Analyze scene every 30 seconds
            self.is_analyzing = False
            print(f"LLaVA model initialized successfully on {self.device}")
        except Exception as e:
            print(f"Error initializing LLaVA: {str(e)}")
            self.model = None
            self.processor = None

    def analyze_frame(self, frame):
        """Analyze frame using LLaVA"""
        if not self.model or not self.processor:
            return None
            
        if self.is_analyzing or time.time() - self.last_analysis < self.analysis_interval:
            return None
        
        self.is_analyzing = True
        try:
            # Convert frame to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Prepare inputs with error handling
            with torch.cuda.amp.autocast() if self.device == 'cuda' else nullcontext():
                inputs = self.processor(
                    "Analyze this security camera footage and describe any potential security concerns or unusual activities.",
                    pil_image,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Generate analysis
                output = self.model.generate(**inputs, max_new_tokens=200)
                analysis = self.processor.decode(output[0], skip_special_tokens=True)
                self.last_analysis = time.time()
                return analysis
        except Exception as e:
            print(f"Error in LLaVA analysis: {str(e)}")
            return None
        finally:
            self.is_analyzing = False
            if self.device == 'cuda':
                torch.cuda.empty_cache()

class SecurityCamera:
    def __init__(self):
        # Initialize components with error handling
        try:
            self.object_detector = ObjectDetector()
            self.face_detector = FaceDetector()
            self.alert_manager = AlertManager()
            self.llava_analyzer = LLaVAAnalyzer()
            
            self.frame_count = 0
            self.last_summary_time = time.time()
            self.running = False
            self.paused = False
            self.last_gc_time = time.time()
            self.gc_interval = 60
            self.processing_interval = 0.1  # Process every 100ms
            self.last_process_time = time.time()
            
            # Start alert processing
            self.alert_manager.start_processing()
            
            print("Security camera initialized successfully")
        except Exception as e:
            print(f"Error initializing security camera: {str(e)}")
            raise
        
    def initialize_video(self):
        """Initialize video capture with error handling"""
        try:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                raise ValueError("Could not open video source")
            
            # Set lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            
            # Set buffer size to 1 to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set frame rate
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            return cap
        except Exception as e:
            print(f"Error initializing video: {str(e)}")
            raise

    def process_frame(self, frame):
        """Process a single frame with error handling"""
        try:
            current_time = time.time()
            if current_time - self.last_process_time < self.processing_interval:
                return frame
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (640, 480))
            
            # Detect objects with error handling
            try:
                detections = self.object_detector.detect_objects(small_frame)
                
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
                    
                    # Add alert for each detection
                    self.alert_manager.add_alert(
                        detection['object'],
                        detection['confidence'],
                        self._get_alert_category(detection),
                        f"Detected {detection['object']} with {detection['confidence']:.2f} confidence"
                    )
                
                # Draw detections
                frame = self.object_detector.draw_detections(frame, detections)
                
            except Exception as e:
                print(f"Error in object detection: {str(e)}")
            
            # Run LLaVA analysis periodically
            try:
                analysis = self.llava_analyzer.analyze_frame(small_frame)
                if analysis:
                    print(f"\nLLaVA Analysis: {analysis}\n")
                    # Add LLaVA analysis as a special alert
                    self.alert_manager.add_alert(
                        "LLaVA Analysis",
                        1.0,
                        "HIGH_PRIORITY",
                        analysis
                    )
            except Exception as e:
                print(f"Error in LLaVA analysis: {str(e)}")
            
            self.last_process_time = current_time
            return frame
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame
    
    def _get_alert_category(self, detection):
        """Determine alert category based on detection"""
        if detection['confidence'] > 0.9:
            return "HIGH_PRIORITY"
        elif detection['confidence'] > 0.7:
            return "MEDIUM_PRIORITY"
        else:
            return "LOW_PRIORITY"
    
    def run(self):
        """Main loop with improved error handling"""
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
                            time.sleep(1)  # Wait before retrying
                            cap = self.initialize_video()
                            continue
                        
                        # Process frame
                        processed_frame = self.process_frame(frame)
                        
                        # Show frame
                        cv2.imshow('Security Camera', processed_frame)
                        
                        # Update performance metrics periodically
                        current_time = time.time()
                        if current_time - self.last_summary_time >= ALERT_SUMMARY_INTERVAL:
                            self.alert_manager.update_performance_metrics(
                                frames_processed=self.frame_count,
                                processing_time=self.processing_interval,
                                false_positives=0,  # Update these based on your detection logic
                                false_negatives=0,
                                accuracy=98.5
                            )
                            self.last_summary_time = current_time
                        
                        # Run garbage collection periodically
                        if current_time - self.last_gc_time >= self.gc_interval:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            self.last_gc_time = current_time
                        
                        self.frame_count += 1
                    
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
                    summary = self.alert_manager.get_alert_summary()
                    if summary:
                        print("\n=== Alert Summary ===")
                        print(f"Total Alerts: {summary['alert_summary']['total_alerts']}")
                        print("==================\n")
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            raise
        
        finally:
            self.running = False
            self.alert_manager.stop_processing()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class nullcontext:
    """Context manager that does nothing"""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def main():
    try:
        camera = SecurityCamera()
        camera.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 
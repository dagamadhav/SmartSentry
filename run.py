import os
import sys
import logging
import torch
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Configure logging
log_dir = os.path.join(project_root, 'data', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import cv2
        import torch
        import numpy as np
        from ultralytics import YOLO
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        logging.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {str(e)}")
        logging.error("Please install all requirements using: pip install -r requirements.txt")
        return False

def check_gpu():
    """Check GPU availability and configuration"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        logging.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        logging.warning("No GPU detected. Running in CPU mode (performance will be limited)")
        return False

def check_directories():
    """Ensure all required directories exist"""
    try:
        directories = [
            os.path.join(project_root, 'data', 'models'),
            os.path.join(project_root, 'data', 'alerts'),
            os.path.join(project_root, 'data', 'logs'),
            os.path.join(project_root, 'known_faces')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Directory checked/created: {directory}")
        
        return True
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        return False

def check_model_files():
    """Check if required model files exist"""
    try:
        model_path = os.path.join(project_root, 'data', 'models', 'yolov8n.pt')
        if not os.path.exists(model_path):
            logging.warning("YOLOv8 model not found. It will be downloaded automatically on first run.")
        return True
    except Exception as e:
        logging.error(f"Error checking model files: {str(e)}")
        return False

def main():
    """Main function to run the application"""
    try:
        logging.info("Starting application initialization...")
        
        # Check system requirements
        if not check_dependencies():
            return
        
        # Check GPU
        has_gpu = check_gpu()
        
        # Check directories
        if not check_directories():
            return
        
        # Check model files
        if not check_model_files():
            return
        
        # Import main application
        from src.main import main as app_main
        
        logging.info("All checks passed. Starting application...")
        
        # Run the application
        app_main()
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        logging.error("Application terminated due to error")
        return 1
    
    finally:
        logging.info("Application shutdown complete")

if __name__ == "__main__":
    sys.exit(main()) 
# Security Camera System with Object Detection

A real-time security camera system that uses YOLO for object detection, face recognition, and alert notifications.

## Features

- Real-time object detection using YOLOv8
- Face recognition for known individuals
- Motion detection
- High-priority alert notifications:
  - Desktop notifications
  - Email notifications
  - Sound alerts
- Clean, modern UI with real-time metrics
- Alert management system
- Performance monitoring

## Prerequisites

- Python 3.8 or higher
- OpenCV
- YOLOv8
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd object_detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights:
```bash
mkdir yolo-Weights
# Download yolov8n.pt from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
# Place it in the yolo-Weights directory
```

4. Set up email notifications (optional):
   - Go to your Google Account settings
   - Enable 2-Step Verification
   - Generate an App Password for Mail
   - Edit `config/notification_config.json` and add your app password

## Project Structure

```
object_detection/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── src/
│   ├── __init__.py
│   ├── alert_manager.py  # Alert management system
│   ├── notification_manager.py  # Notification system
│   └── config/
│       └── notification_config.json  # Notification settings
├── known_faces/          # Directory for known face images
└── yolo-Weights/         # YOLO model weights
    └── yolov8n.pt
```

## Usage

1. Add known faces:
   - Place images of known people in the `known_faces` directory
   - Name the files as `person_name.jpg` (e.g., `john.jpg`)

2. Run the application:
```bash
python app.py
```

3. Controls:
   - Press 'q' to quit
   - Press 's' to show alert statistics

## Alert Categories

- HIGH_PRIORITY: person, car, truck, bus
- MEDIUM_PRIORITY: bicycle, motorbike, dog, cat
- LOW_PRIORITY: bird, chair, bottle
- IGNORED: tree, leaves, clouds

## Notification System

The system sends notifications when:
- High-priority objects are detected
- Detections persist for more than 2 seconds
- Known persons are recognized

Notification methods:
1. Desktop notifications (Windows)
2. Email notifications (Gmail)
3. System sound alerts

## Configuration

### Email Notifications
Edit `config/notification_config.json`:
```json
{
    "email": {
        "enabled": true,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "your-email@gmail.com",
        "sender_password": "your-app-password",
        "recipient_email": "your-email@gmail.com"
    },
    "notification_sound": true,
    "desktop_notification": true
}
```

## Performance

- Minimum confidence threshold: 0.5
- Alert cooldown: 120 seconds
- Motion detection threshold: 5000
- Face recognition interval: 30 frames

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

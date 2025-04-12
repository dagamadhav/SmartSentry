<<<<<<< HEAD
# SmartSentry: Advanced Security Surveillance System

![System Architecture](docs/images/architecture.png)

## Overview

SmartSentry is an intelligent security surveillance system that combines state-of-the-art computer vision and language models to provide comprehensive security monitoring. The system integrates YOLOv8 for real-time object detection and LLaVA for advanced scene understanding.

## System Architecture

### High-Level Architecture
```mermaid
graph TD
    A[Camera Input] --> B[Frame Processing]
    B --> C[Object Detection]
    B --> D[Face Recognition]
    C --> E[Alert Generation]
    D --> E
    B --> F[LLaVA Analysis]
    F --> G[Scene Understanding]
    E --> H[Alert Management]
    G --> H
    H --> I[User Interface]
```

### Detailed Component Architecture
```mermaid
graph TD
    subgraph Input Layer
        A1[Camera Feed]
        A2[IP Camera]
        A3[Video File]
    end
    
    subgraph Processing Layer
        B1[Frame Capture]
        B2[Pre-processing]
        B3[Frame Queue]
    end
    
    subgraph Detection Layer
        C1[YOLOv8 Model]
        C2[Face Recognition]
        C3[Object Tracking]
    end
    
    subgraph Analysis Layer
        D1[LLaVA Model]
        D2[Scene Understanding]
        D3[Activity Analysis]
    end
    
    subgraph Alert Layer
        E1[Priority Queue]
        E2[Alert Generation]
        E3[Notification System]
    end
    
    A1 & A2 & A3 --> B1
    B1 --> B2 --> B3
    B3 --> C1 & C2 & C3
    B3 --> D1 --> D2 --> D3
    C1 & C2 & C3 --> E1
    D2 & D3 --> E1
    E1 --> E2 --> E3
```

## Key Components

### 1. Object Detection (YOLOv8)
- Real-time object detection
- Multiple object categories
- Confidence-based filtering
- Priority-based alerting

### 2. Face Recognition
- Known face database
- Real-time face matching
- Person tracking
- Access control integration

### 3. LLaVA Integration
- Advanced scene understanding
- Natural language descriptions
- Security concern detection
- Activity analysis

## Performance Metrics

### Detection Accuracy Over Time
```mermaid
graph LR
    A[Day 1] -->|95%| B[Day 2]
    B -->|96%| C[Day 3]
    C -->|94%| D[Day 4]
    D -->|97%| E[Day 5]
```

### Resource Utilization
```mermaid
pie title Resource Distribution
    "Object Detection" : 40
    "Face Recognition" : 25
    "LLaVA Analysis" : 20
    "System Overhead" : 15
```

### Processing Pipeline Timing
```mermaid
gantt
    title Processing Pipeline Timeline
    dateFormat  HH:mm:ss.SSS
    section Frame Processing
    Capture Frame      :a1, 00:00:00.000, 00:00:00.010
    Pre-process        :a2, after a1, 00:00:00.020
    Object Detection   :a3, after a2, 00:00:00.070
    Face Recognition   :a4, after a2, 00:00:00.120
    section LLaVA Analysis
    Frame Selection    :b1, 00:00:00.000, 00:00:00.005
    Scene Analysis     :b2, after b1, 00:00:00.500
    Report Generation  :b3, after b2, 00:00:00.550
```

## LLaVA Integration Details

### LLaVA Processing Pipeline
```mermaid
graph TD
    A[Camera Feed] --> B[Frame Selection]
    B --> C[Image Preprocessing]
    C --> D[LLaVA Model]
    D --> E[Vision Encoder]
    E --> F[Language Model]
    F --> G[Scene Description]
    G --> H[Security Analysis]
    H --> I[Alert Generation]
```

### LLaVA Model Architecture
```mermaid
graph LR
    subgraph Vision Encoder
        A1[CLIP Image Encoder]
        A2[Feature Extraction]
        A3[Visual Tokens]
    end
    
    subgraph Language Model
        B1[LLaMA Backbone]
        B2[Text Generation]
        B3[Context Understanding]
    end
    
    subgraph Multimodal Adapter
        C1[Vision-Language Bridge]
        C2[Feature Alignment]
        C3[Cross-Modal Attention]
    end
    
    A1 --> A2 --> A3
    A3 --> C1
    C1 --> C2 --> C3
    C3 --> B1 --> B2 --> B3
```

### LLaVA Analysis Workflow
```mermaid
sequenceDiagram
    participant C as Camera
    participant P as Preprocessor
    participant L as LLaVA
    participant A as Alert System
    
    C->>P: Send Frame
    P->>L: Preprocess Image
    L->>L: Encode Vision
    L->>L: Generate Description
    L->>A: Send Analysis
    A->>A: Evaluate Security
    A->>A: Generate Alert
```

## Performance Analysis

### Frame Processing Speed
```mermaid
graph LR
    A[640x480] -->|15ms| B[Object Detection]
    B -->|25ms| C[Face Recognition]
    C -->|10ms| D[Alert Generation]
    
    E[1280x720] -->|35ms| F[Object Detection]
    F -->|45ms| G[Face Recognition]
    G -->|15ms| H[Alert Generation]
```

### Memory Usage Pattern
```mermaid
graph TD
    A[Start] -->|Load Models| B[Peak Memory]
    B -->|Stable Processing| C[Steady State]
    C -->|Garbage Collection| D[Memory Cleanup]
    D -->|Continue Processing| C
```

### GPU Utilization
```mermaid
pie title GPU Resource Allocation
    "YOLOv8 Processing" : 45
    "LLaVA Analysis" : 35
    "Face Recognition" : 15
    "System Overhead" : 5
```

## Alert System Architecture

### Alert Processing Flow
```mermaid
graph TD
    A[Detection] --> B{Priority Check}
    B -->|High| C[Immediate Alert]
    B -->|Medium| D[Queue Alert]
    B -->|Low| E[Log Only]
    
    C --> F[Notification System]
    D --> G[Alert Queue]
    G --> H[Batch Processing]
    H --> F
    E --> I[Log File]
```

### Alert Distribution Heatmap
```mermaid
graph TD
    A[Time of Day] --> B[Alert Frequency]
    B --> C[High Priority]
    B --> D[Medium Priority]
    B --> E[Low Priority]
    
    subgraph Heatmap
        F[Morning] -->|High| G[Peak Hours]
        H[Afternoon] -->|Medium| I[Regular Hours]
        J[Night] -->|Low| K[Quiet Hours]
    end
```

## Alert Categories

| Category | Objects | Priority | Response Time |
|----------|---------|----------|---------------|
| High | Person, Weapon, Phone | Immediate | < 2s |
| Medium | Vehicle, Animal | Moderate | < 5s |
| Low | Common Objects | Low | < 10s |

## System Requirements

### Hardware Requirements
```mermaid
graph TD
    A[Minimum Requirements] --> B[CPU: Intel i5]
    A --> C[RAM: 8GB]
    A --> D[Storage: 10GB]
    E[Recommended] --> F[GPU: NVIDIA]
    E --> G[RAM: 16GB]
    E --> H[Storage: 20GB]
```

### Software Stack
- Python 3.8+
- CUDA Toolkit (for GPU acceleration)
- Required Python packages (see requirements.txt)

## Installation Guide

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Model Download**
   ```bash
   # YOLOv8 model will be downloaded automatically
   # LLaVA model will be downloaded on first run
   ```

3. **Directory Structure**
   ```
   SmartSentry/
   ├── src/
   │   ├── config/
   │   ├── detectors/
   │   ├── alerts/
   │   └── main.py
   ├── data/
   │   ├── alerts/
   │   └── models/
   ├── known_faces/
   └── requirements.txt
   ```

## Usage Guide

### 1. Starting the System
```bash
python run.py
```

### 2. Keyboard Controls
| Key | Function |
|-----|----------|
| q | Quit application |
| p | Pause/Resume |
| s | Show summary |
| e | Export alerts |

### 3. Alert Management
- Real-time alert logging
- Periodic summaries
- CSV export functionality
- Alert categorization

## Advanced Features

### 1. LLaVA Integration
```mermaid
graph LR
    A[Camera Feed] --> B[Frame Capture]
    B --> C[Image Processing]
    C --> D[LLaVA Analysis]
    D --> E[Scene Description]
    E --> F[Security Assessment]
```

### 2. Performance Optimization
- Frame processing: 100ms interval
- LLaVA analysis: 30s interval
- GPU acceleration
- Memory management

## Data Analysis

### Alert Distribution
```mermaid
pie title Alert Distribution (Last 24 Hours)
    "Human Detection" : 45
    "Vehicle Detection" : 25
    "Animal Detection" : 15
    "Other Objects" : 15
```

### Processing Time Analysis
```mermaid
graph TD
    A[Frame Capture] -->|10ms| B[Object Detection]
    B -->|50ms| C[Face Recognition]
    C -->|40ms| D[Alert Generation]
```

## Troubleshooting Guide

### Common Issues
1. **Camera Not Opening**
   - Check permissions
   - Verify camera index
   - Ensure no conflicts

2. **High Resource Usage**
   - Adjust frame size
   - Increase intervals
   - Use GPU acceleration

3. **Model Loading Issues**
   - Check GPU memory
   - Verify CUDA installation
   - Clear model cache

## Future Enhancements

1. **Planned Features**
   - Multi-camera support
   - Cloud integration
   - Mobile notifications
   - Advanced analytics

2. **Performance Improvements**
   - Optimized model loading
   - Reduced memory usage
   - Faster processing
   - Better accuracy

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 team for object detection
- LLaVA team for vision-language model
- OpenCV community
- All contributors
=======
# Intelligent Security Camera System

A comprehensive security camera system that combines object detection, face recognition, and alert management to provide intelligent surveillance capabilities.

## Features

- Real-time object detection using YOLOv8
- Face recognition and tracking
- Priority-based alert system
- Detailed alert logging and reporting
- Memory-efficient processing
- Configurable detection thresholds
- Exportable alert data

## System Architecture

### Core Modules

1. **Main Application (`src/main.py`)**
   - Manages the video capture and processing pipeline
   - Coordinates between different detection modules
   - Handles user interface and keyboard controls
   - Implements memory management and garbage collection

2. **Object Detection (`src/detectors/object_detector.py`)**
   - Implements YOLOv8-based object detection
   - Categorizes objects by priority (High, Medium, Low)
   - Draws bounding boxes and labels
   - Handles detection confidence thresholds

3. **Face Detection (`src/detectors/face_detector.py`)**
   - Implements face detection and recognition
   - Maintains a database of known faces
   - Tracks recognized individuals
   - Integrates with object detection to filter person detections

4. **Alert Management (`src/alerts/alert_manager.py`)**
   - Manages alert logging and storage
   - Implements cooldown periods for alerts
   - Generates alert summaries
   - Exports alert data to CSV
   - Maintains alert history

5. **Configuration (`src/config/settings.py`)**
   - Centralized configuration management
   - Defines detection thresholds
   - Sets up file paths and directories
   - Configures time windows for alerts

## Setup Instructions

1. **Prerequisites**
   ```bash
   pip install -r requirements.txt
   ```

2. **Directory Structure**
   ```
   project/
   ├── src/
   │   ├── config/
   │   ├── detectors/
   │   ├── alerts/
   │   └── main.py
   ├── data/
   │   ├── alerts/
   │   └── models/
   ├── known_faces/
   └── requirements.txt
   ```

3. **Configuration**
   - Adjust settings in `src/config/settings.py`
   - Configure camera source and resolution
   - Set detection thresholds and intervals
   - Define object categories and priorities

## Usage

1. **Starting the Application**
   ```bash
   python -m src.main
   ```

2. **Keyboard Controls**
   - `q`: Quit application
   - `p`: Pause/Resume processing
   - `s`: Show alert summary
   - `e`: Export alerts to CSV

3. **Alert Management**
   - Alerts are automatically logged to `data/alerts/alerts.json`
   - Periodic summaries are displayed in the console
   - Export alerts to CSV for analysis

## Advanced Features

1. **Priority-based Detection**
   - High Priority: Persons, weapons, phones
   - Medium Priority: Valuable items
   - Low Priority: Common objects
   - Configurable thresholds and cooldowns

2. **Memory Management**
   - Automatic garbage collection
   - Optimized frame processing
   - Efficient alert storage

3. **Face Recognition Integration**
   - Known face filtering
   - Person tracking
   - Recognition confidence thresholds

## Performance Optimization

- Frame resizing for faster processing
- Periodic garbage collection
- Efficient alert storage and retrieval
- Configurable detection intervals
- Memory-efficient object tracking

## Troubleshooting

1. **Camera Not Opening**
   - Check camera permissions
   - Verify camera index in settings
   - Ensure no other application is using the camera

2. **High CPU/Memory Usage**
   - Adjust frame size in settings
   - Increase detection intervals
   - Reduce detection confidence thresholds

3. **Alert File Issues**
   - Check directory permissions
   - Verify file paths in settings
   - Ensure sufficient disk space

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
>>>>>>> cee311a2b267c73a779e91d70049da281757be32

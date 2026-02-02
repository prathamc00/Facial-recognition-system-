# Facial Recognition System

A comprehensive machine learning-based facial recognition system with liveness detection, attendance tracking, and real-time performance monitoring. Built with Python, TensorFlow, and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### Core Functionality
- **Real-time Face Detection** - Fast and accurate face detection using OpenCV Haar Cascades
- **Face Recognition** - Deep learning-based recognition using MobileNetV2 transfer learning
- **Multi-Person Registration** - Easy registration system for multiple people
- **SQLite Database** - Efficient storage of face data with quality metrics

### Advanced Features
- **Liveness Detection** - Anti-spoofing with eye blink and texture analysis
- **Attendance Tracking** - Automatic attendance logging with Excel export
- **Performance Monitoring** - Real-time FPS, accuracy, and timing metrics
- **Data Augmentation** - Automatic training data enhancement for better accuracy
- **Smart Training** - Early stopping and adaptive learning rate

### User Interface
- **Modern GUI** - Intuitive tkinter-based interface
- **Live Video Feed** - Real-time webcam preview with annotations
- **Multiple Modes** - Detection, Recognition, and Registration modes
- **Visual Metrics** - Live performance statistics display

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows OS (for automated setup script)

### Installation

#### Automated Setup (Windows)

1. **Clone or download this repository**

2. **Run the setup script:**
   ```batch
   setup_windows.bat
   ```
   This will:
   - Check for Python installation
   - Install all dependencies
   - Create necessary directories
   - Initialize the database

3. **Launch the application:**
   ```batch
   python src/gui_app.py
   ```

#### Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create directories:**
   ```bash
   mkdir data models logs
   ```

3. **Run the application:**
   ```bash
   python src/gui_app.py
   ```

## Usage Guide

### 1. Register Faces

1. Click **"Registration"** button
2. Enter the person's name when prompted
3. Look at the camera while the system collects 10 face images
4. Click **"Save Faces"** to save to database

### 2. Train the Model

1. After registering at least 2 people, click **"Train Model"**
2. Wait for training to complete (shows progress dialog)
3. Model is automatically saved to `models/` directory

### 3. Recognize Faces

1. Click **"Recognition"** mode
2. The system will automatically recognize and label faces
3. Attendance is logged automatically with timestamps

### 4. View Attendance

- Attendance logs are saved in `logs/attendance_YYYY-MM-DD.csv`
- Use the attendance logger to export to Excel format
- Duplicate entries within 30 seconds are automatically filtered

## Project Structure

```
Facial-recognition-system/
├── data/                      # Face images and database
│   └── face_database.db      # SQLite database
├── models/                    # Trained models
│   └── face_recognition_model.h5
├── logs/                      # Attendance and performance logs
│   ├── attendance_*.csv
│   └── recognition_metrics.csv
├── src/                       # Source code
│   ├── gui_app.py            # Main GUI application
│   ├── face_detection.py     # Face detection module
│   ├── face_recognition.py   # Face recognition module
│   ├── face_database.py      # Database operations
│   ├── liveness_detection.py # Anti-spoofing
│   ├── attendance_logger.py  # Attendance tracking
│   ├── performance_logger.py # Performance metrics
│   └── data_augmentation.py  # Data augmentation
├── requirements.txt           # Windows dependencies
├── requirements_rpi.txt       # Raspberry Pi dependencies
├── config.json               # Configuration settings
├── setup_windows.bat         # Automated setup script
└── README.md                 # This file
```

## Configuration

Edit `config.json` to customize:

```json
{
  "recognition": {
    "confidence_threshold": 0.75,
    "min_face_size": 80
  },
  "training": {
    "epochs": 20,
    "use_augmentation": true,
    "augmentation_factor": 3
  },
  "liveness": {
    "enabled": true
  }
}
```

## Advanced Features

### Data Augmentation

The system automatically augments training data with:
- Random rotation (±15°)
- Brightness adjustment
- Contrast variation
- Gaussian noise
- Horizontal flipping

Enable in training:
```python
recognizer.train(face_images, labels, use_augmentation=True, augmentation_factor=3)
```

### Liveness Detection

Prevents spoofing with:
- **Eye Blink Detection** - Tracks eye aspect ratio
- **Motion Detection** - Analyzes frame-to-frame changes
- **Texture Analysis** - Detects print/screen artifacts

Toggle in GUI or code:
```python
liveness_detector.check_liveness(frame, face_rect)
```

### Performance Monitoring

Track system performance:
```python
logger = PerformanceLogger()
metrics = logger.get_metrics()
# Returns: FPS, detection time, recognition time, etc.
```

### Attendance Export

Export attendance to Excel:
```python
attendance_logger.export_to_excel(
    start_date='2026-02-01',
    end_date='2026-02-28',
    output_file='february_attendance.xlsx'
)
```

## Performance

Typical performance on a modern PC:
- **FPS**: 25-30 FPS
- **Detection Time**: 15-25ms per frame
- **Recognition Time**: 30-50ms per face
- **Training Time**: 2-5 minutes (depends on dataset size)

## Troubleshooting

### Camera Not Working
```python
# Try different camera index
camera = cv2.VideoCapture(1)  # 0, 1, 2, etc.
```

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Low Recognition Accuracy
- Add more training images (10-20 per person)
- Ensure good lighting conditions
- Retrain with data augmentation enabled
- Check face image quality in database

### TensorFlow Warnings
- These are normal and can be ignored
- GPU support is optional for better performance

## Security Features

- **Liveness Detection** - Prevents photo/video spoofing
- **Confidence Thresholds** - Reject low-confidence recognitions
- **Quality Metrics** - Only save high-quality face images
- **Duplicate Prevention** - Avoid logging same person multiple times

## Raspberry Pi Support

For Raspberry Pi deployment, use:
```bash
pip install -r requirements_rpi.txt
```

Optimizations for Pi:
- Reduced resolution (640x480)
- Lower FPS target (10-15 FPS)
- ARM-optimized TensorFlow
- Hardware camera support

## API Reference

### FaceRecognizer

```python
from face_recognition import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.build_model(num_classes=5)
history = recognizer.train(face_images, labels, epochs=20)
label, confidence = recognizer.recognize(face_image)
recognizer.save_model('model.h5')
```

### FaceDatabase

```python
from face_database import FaceDatabase

db = FaceDatabase('faces.db')
person_id = db.add_person('John Doe')
db.add_face(person_id, face_image, face_encoding)
people = db.get_all_people()
```

### AttendanceLogger

```python
from attendance_logger import AttendanceLogger

logger = AttendanceLogger()
logger.log_attendance('John Doe', confidence=0.95)
stats = logger.get_statistics()
logger.export_to_excel(output_file='attendance.xlsx')
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **OpenCV** - Computer vision library
- **TensorFlow/Keras** - Deep learning framework
- **MobileNetV2** - Transfer learning base model
- **dlib** - Facial landmark detection

## Support

For questions or support, please open an issue on the repository.

---

**Made with love by Pratham**
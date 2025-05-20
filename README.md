# Face Recognition System

A machine learning-based face recognition system that can detect and recognize faces in images and video streams using a webcam interface.

## Features

- Face detection in images and video streams using OpenCV's Haar Cascade classifier
- Face recognition and identification using a deep learning model based on MobileNetV2
- Real-time webcam integration for face detection and recognition
- SQLite database for storing facial data and person information
- Simple keyboard interface for interaction

## Project Structure

```
facial_detection/
├── data/                  # Directory for storing training data and face database
├── models/                # Directory for storing trained models
├── src/                   # Source code
│   ├── face_detection.py  # Face detection module
│   ├── face_recognition.py # Face recognition module
│   ├── database.py        # Database operations
│   └── main.py            # Main application
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

### General Installation
1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Raspberry Pi Specific Setup
1. Update your Raspberry Pi:
   ```
   sudo apt-get update
   sudo apt-get upgrade
   ```

2. Install system dependencies:
   ```
   sudo apt-get install -y python3-pip python3-dev libatlas-base-dev
   sudo apt-get install -y libjasper-dev libqtgui4 libqt4-test libhdf5-dev
   ```

3. Install Python dependencies optimized for Raspberry Pi:
   ```
   pip3 install opencv-python-headless  # Optimized OpenCV build
   pip3 install tensorflow-aarch64      # ARM-optimized TensorFlow
   pip3 install picamera2               # For Raspberry Pi camera
   ```

### Performance Optimization for Raspberry Pi
- Use `picamera2` instead of OpenCV for video capture
- Reduce resolution to 640x480 for faster processing
- Consider running inference at 2-3 FPS for better performance
- Enable OpenCV GPU acceleration if available

## Usage

prat### Detailed Setup Instructions
1. **Setup**:
   - Install Python 3.8 or higher
   - Install dependencies: `pip install -r requirements.txt`
   - Create a database: `python src/database.py --init`
   - Ensure webcam is properly connected and working

### Face Registration Process
2. **Register Faces**:
   - Create a folder for each person in `data/` directory (e.g., `data/john_doe/`)
   - Place at least 10 clear face images per person in their respective folder
   - Images should show different angles and lighting conditions
   - Run registration: `python src/main.py --register`
   - Follow on-screen prompts to complete registration

### Model Training
3. **Train Model**:
   - Ensure you have registered at least 2 different people
   - Train the recognition model: `python src/main.py --train`
   - Training progress will be displayed in console
   - Model will be saved to `models/` directory

### Recognition Modes
4. **Recognize Faces**:
   - For image recognition: `python src/main.py --recognize --image path/to/image.jpg`
   - For webcam recognition: `python src/main.py --live`
   - Recognition confidence threshold is set at 85% by default

### Additional Options
5. **Additional Options**:
   - View help: `python src/main.py --help`
   - Clear database: `python src/database.py --clear`
   - View registered persons: `python src/database.py --list`

### Troubleshooting
- If webcam doesn't work:
  - Check device permissions
  - Try different camera index (e.g., `--camera 1`)
- If recognition accuracy is low:
  - Add more training images
  - Ensure good lighting conditions
  - Retrain the model

### Commands

The application supports the following keyboard commands:

- `q` - Quit the application
- `r` - Switch to registration mode (you'll be prompted to enter a person's name)
- `f` - Switch to recognition mode
- `d` - Switch to detection mode
- `s` - Save collected faces (in registration mode)
- `t` - Train recognition model

### Workflow

1. **Register Faces**:
   - Press `r` and enter a person's name
   - The system will collect 10 face images
   - Press `s` to save the collected faces

2. **Train the Model**:
   - After registering faces for multiple people, press `t` to train the recognition model
   - The model will be saved to the `models` directory

3. **Recognize Faces**:
   - Press `f` to switch to recognition mode
   - The system will identify faces in the webcam feed and display names with confidence scores

## Dependencies

- OpenCV - For image processing and face detection
- NumPy - For numerical operations
- TensorFlow/Keras - For deep learning models
- SQLite - For database operations
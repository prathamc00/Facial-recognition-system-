# Enhanced Facial Recognition System

This is an enhanced version of the original facial recognition system with several improvements for better usability, security, and performance tracking.

## New Features

### 1. User-Friendly GUI Interface
- Replaced keyboard commands with intuitive buttons and controls
- Real-time video feed display with status indicators
- Easy-to-use registration and recognition workflows
- Visual feedback for face detection and recognition

### 2. Liveness Detection
- Anti-spoofing technology to prevent photo and video attacks
- Detects eye blinks and facial movements to verify a live person
- Configurable security settings with toggle option
- Visual indicators for liveness verification status

### 3. Performance Metrics and Logging
- Tracks recognition accuracy over time
- Measures and displays confidence scores
- Records recognition speed and success rates
- Maintains detailed logs for system performance analysis

### 4. Multi-Face Recognition
- Ability to detect and recognize multiple faces simultaneously
- Individual tracking of each detected face
- Separate liveness verification for each face

## Installation

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. For liveness detection, download the shape predictor file:
   - Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract and place in the `models` directory

## Usage

### Starting the Application

Run the enhanced application with:
```
python src/enhanced_main.py
```

### Workflow

1. **Register Faces**:
   - Click the "Registration" button
   - Enter the person's name when prompted
   - The system will automatically collect face samples
   - Click "Save Faces" when 10 samples are collected

2. **Train the Model**:
   - After registering faces for multiple people, click "Train Model"
   - Wait for the training to complete
   - The model will be saved automatically

3. **Recognize Faces**:
   - Click "Recognition" to switch to recognition mode
   - The system will identify faces in the webcam feed
   - Names and confidence scores will be displayed
   - Liveness verification will prevent photo spoofing

### Security Settings

- Toggle liveness detection on/off using the checkbox
- View performance metrics in real-time
- Reset metrics to start fresh measurements

## Performance Monitoring

The system maintains logs in the `logs` directory:
- Recognition attempts with timestamps and confidence scores
- Training metrics for model evaluation
- Success rates and average recognition times

## Technical Details

- Face detection using OpenCV Haar cascades
- Face recognition using deep learning with MobileNetV2
- Liveness detection with eye aspect ratio and motion analysis
- SQLite database for storing face data and encodings
- Tkinter-based GUI for cross-platform compatibility

## Future Improvements

- Emotion detection capabilities
- Age and gender estimation
- Integration with access control systems
- Cloud synchronization for distributed recognition
- Mobile application support
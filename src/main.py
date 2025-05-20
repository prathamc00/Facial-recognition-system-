import os
import cv2
import numpy as np
import time

from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from face_database import FaceDatabase

class FacialRecognitionApp:
    def __init__(self):
        """
        Initialize the facial recognition application.
        """
        # Initialize components
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.db = FaceDatabase("../data/faces.db")  # Initialize database with path
        
        # Check if model exists and load it
        model_path = "../models/face_recognition_model.h5"
        if os.path.exists(model_path):
            self.recognizer.load_model(model_path)
            print(f"Loaded face recognition model from {model_path}")
        else:
            print("No pre-trained model found. Please follow these steps:\n1. Press 'r' to register new people\n2. Collect at least 10 face samples per person\n3. Press 's' to save collected faces\n4. Press 't' to train the model\n")
        
        # Initialize camera
        self.camera = None
        
        # Application state
        self.is_running = False
        self.current_mode = "detection"  # 'detection', 'registration', 'recognition'
        self.current_person_name = ""
        self.current_person_id = None
        self.collected_faces = []
        
    def start_camera(self, camera_id=0):
        """
        Start the camera capture.
        
        Args:
            camera_id (int): Camera device ID
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        self.camera = cv2.VideoCapture(camera_id)
        return self.camera.isOpened()
    
    def stop_camera(self):
        """
        Stop the camera capture.
        """
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def register_new_person(self, name):
        """
        Start the registration process for a new person.
        
        Args:
            name (str): Name of the person to register
        """
        self.current_mode = "registration"
        self.current_person_name = name
        self.current_person_id = self.db.add_person(name)
        self.collected_faces = []
        print(f"Started registration for {name} (ID: {self.current_person_id})")
    
    def collect_face(self, face_image):
        """
        Collect a face image during registration.
        
        Args:
            face_image (numpy.ndarray): Face image to collect
        """
        if self.current_mode == "registration" and self.current_person_id is not None:
            self.collected_faces.append(face_image)
            print(f"Collected face {len(self.collected_faces)} for {self.current_person_name}")
    
    def complete_registration(self):
        """
        Complete the registration process by saving collected faces to the database.
        
        Returns:
            bool: True if registration was successful, False otherwise
        """
        if self.current_mode != "registration" or not self.collected_faces:
            print("No faces collected for registration")
            return False
        
        # Save faces to database
        for face_image in self.collected_faces:
            # Generate face encoding
            face_encoding = self.recognizer.extract_features(face_image)
            
            # Add to database
            self.db.add_face(self.current_person_id, face_image, face_encoding)
        
        print(f"Registration complete for {self.current_person_name} with {len(self.collected_faces)} faces")
        
        # Reset registration state
        self.current_mode = "detection"
        self.current_person_name = ""
        self.current_person_id = None
        self.collected_faces = []
        
        return True
    
    def train_model(self):
        """
        Train the face recognition model using faces in the database.
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        print("Starting model training...")
        
        # Get all people from database
        people = self.db.get_all_people()
        if not people:
            print("No people in database. Please register people first by pressing 'r'.")
            return False
        
        # Collect face images and labels
        face_images = []
        labels = []
        
        for person in people:
            person_faces = self.db.get_person_faces(person['id'])
            for face_data in person_faces:
                if face_data['face_encoding'] is not None:
                    face_images.append(face_data['face_image'])
                    labels.append(person['name'])
        
        if not face_images:
            print("No face data available for training. Please collect face samples by registering people first.")
            return False
        
        # Train the model
        self.recognizer.train(face_images, labels)
        
        # Save the model
        os.makedirs("../models", exist_ok=True)
        model_path = "../models/face_recognition_model.h5"
        self.recognizer.save_model(model_path)
        
        print(f"Model trained and saved to {model_path}")
        return True
    
    def recognize_face(self, face_image):
        """
        Recognize a face in the image.
        
        Args:
            face_image (numpy.ndarray): Face image to recognize
            
        Returns:
            tuple: (person_name, confidence) or (None, 0) if not recognized
        """
        if self.recognizer.model is None:
            print("No model loaded. Train or load a model first.")
            return None, 0
        
        return self.recognizer.recognize(face_image)
    
    def process_frame(self, frame):
        """
        Process a video frame based on the current mode.
        
        Args:
            frame (numpy.ndarray): Video frame to process
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        # Detect faces in the frame
        faces = self.detector.detect_faces(frame)
        
        # Draw rectangles around faces
        result_frame = self.detector.draw_faces(frame, faces)
        
        # Extract face images
        if faces is not None and len(faces) > 0:
            try:
                face_images = self.detector.extract_faces(frame, faces)
                
                # Process based on current mode
                if self.current_mode == "registration" and len(self.collected_faces) < 10:
                    # In registration mode, collect faces
                    if len(face_images) == 1:  # Only collect if there's exactly one face
                        self.collect_face(face_images[0])
                        # Add text to indicate face collection
                        cv2.putText(result_frame, f"Collecting face {len(self.collected_faces)}/10", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif self.current_mode == "recognition":
                    # In recognition mode, recognize faces
                    for i, face_image in enumerate(face_images):
                        if i < len(faces):  # Make sure we don't go out of bounds
                            person_name, confidence = self.recognize_face(face_image)
                            
                            if person_name and confidence > 0.5:
                                # Add text with recognized name and confidence
                                x, y, w, h = faces[i]
                                label = f"{person_name} ({confidence:.2f})"
                                cv2.putText(result_frame, label, (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing faces: {str(e)}")
        
        # Add mode indicator
        cv2.putText(result_frame, f"Mode: {self.current_mode}", 
                    (10, result_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return result_frame
    
    def run(self):
        """
        Run the main application loop.
        """
        if not self.start_camera():
            print("Failed to start camera")
            return
        
        self.is_running = True
        self.current_mode = "detection"
        
        print("\nFacial Recognition System")
        print("=======================")
        print("Commands:")
        print("  'q' - Quit")
        print("  'r' - Switch to registration mode")
        print("  'f' - Switch to recognition mode")
        print("  'd' - Switch to detection mode")
        print("  's' - Save collected faces (in registration mode)")
        print("  't' - Train recognition model")
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to  capture frame")
                break
            
            # Process the frame
            frame = cv2.flip(frame, 1)  # Horizontal flip to mirror the display
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow("Facial Recognition System", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                self.is_running = False
            elif key == ord('r'):
                # Registration mode
                name = input("Enter person's name: ")
                if name:
                    self.register_new_person(name)
            elif key == ord('f'):
                # Recognition mode
                self.current_mode = "recognition"
                print("Switched to recognition mode")
            elif key == ord('d'):
                # Detection mode
                self.current_mode = "detection"
                print("Switched to detection mode")
            elif key == ord('s'):
                # Save collected faces
                if self.current_mode == "registration":
                    self.complete_registration()
            elif key == ord('t'):
                # Train model
                self.train_model()
        
        # Clean up
        self.stop_camera()
        cv2.destroyAllWindows()
        if hasattr(self, 'db') and self.db:
            self.db.close()

if __name__ == "__main__":
    app = FacialRecognitionApp()
    app.run()
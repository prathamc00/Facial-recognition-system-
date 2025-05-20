import os
import cv2
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import threading

from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from face_database import FaceDatabase
from performance_logger import PerformanceLogger
from liveness_detection import LivenessDetector

class FacialRecognitionGUI:
    def __init__(self, root):
        """
        Initialize the facial recognition GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize components
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.db = FaceDatabase("../data/faces.db")
        self.logger = PerformanceLogger()
        self.liveness_detector = LivenessDetector()
        
        # Security settings
        self.liveness_check_enabled = True  # Enable liveness detection by default
        
        # Check if model exists and load it
        model_path = "../models/face_recognition_model.h5"
        if os.path.exists(model_path):
            self.recognizer.load_model(model_path)
            messagebox.showinfo("Model Loaded", f"Loaded face recognition model from {model_path}")
        else:
            messagebox.showinfo("No Model Found", "No pre-trained model found. Please register faces and train the model.")
        
        # Application state
        self.is_running = False
        self.current_mode = "detection"  # 'detection', 'registration', 'recognition'
        self.current_person_name = ""
        self.current_person_id = None
        self.collected_faces = []
        self.camera = None
        self.camera_thread = None
        self.stop_event = threading.Event()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start camera
        self.start_camera()
    
    def create_widgets(self):
        """
        Create GUI widgets.
        """
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (video feed)
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel (controls)
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(mode_frame, text="Detection", command=lambda: self.set_mode("detection")).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(mode_frame, text="Recognition", command=lambda: self.set_mode("recognition")).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(mode_frame, text="Registration", command=self.start_registration).pack(fill=tk.X, padx=5, pady=2)
        
        # Actions
        action_frame = ttk.LabelFrame(control_frame, text="Actions")
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Save Faces", command=self.save_faces).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Train Model", command=self.train_model).pack(fill=tk.X, padx=5, pady=2)
        
        # Status
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.mode_label = ttk.Label(status_frame, text="Mode: Detection")
        self.mode_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.faces_label = ttk.Label(status_frame, text="Collected Faces: 0/10")
        self.faces_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Security settings
        security_frame = ttk.LabelFrame(control_frame, text="Security Settings")
        security_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Liveness detection toggle
        self.liveness_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(security_frame, text="Enable Liveness Detection", 
                       variable=self.liveness_var, 
                       command=self.toggle_liveness_detection).pack(fill=tk.X, padx=5, pady=2)
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(control_frame, text="Performance Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.recognition_rate_label = ttk.Label(metrics_frame, text="Success Rate: 0%")
        self.recognition_rate_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.avg_confidence_label = ttk.Label(metrics_frame, text="Avg. Confidence: 0.00")
        self.avg_confidence_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.avg_time_label = ttk.Label(metrics_frame, text="Avg. Time: 0.00 ms")
        self.avg_time_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Reset metrics button
        ttk.Button(metrics_frame, text="Reset Metrics", command=self.reset_metrics).pack(fill=tk.X, padx=5, pady=2)
    
    def start_camera(self, camera_id=0):
        """
        Start the camera in a separate thread.
        """
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Failed to start camera")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.camera_thread = threading.Thread(target=self.update_frame)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def update_frame(self):
        """
        Update the video frame continuously.
        """
        while self.is_running and not self.stop_event.is_set():
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame based on current mode
            start_time = time.time()
            processed_frame = self.process_frame(frame)
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Convert to tkinter format
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.config(width=processed_frame.shape[1], height=processed_frame.shape[0])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img  # Keep a reference
            
            # Update metrics display periodically
            if self.current_mode == "recognition" and time.time() % 5 < 0.1:  # Update roughly every 5 seconds
                self.update_metrics_display()
    
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
                        # Update faces label
                        self.root.after(0, lambda: self.faces_label.config(text=f"Collected Faces: {len(self.collected_faces)}/10"))
                
                elif self.current_mode == "recognition":
                    # In recognition mode, recognize faces
                    for i, face_image in enumerate(face_images):
                        if i < len(faces):  # Make sure we don't go out of bounds
                            x, y, w, h = faces[i]
                            
                            # Check liveness if enabled
                            is_live = True
                            liveness_score = 1.0
                            if self.liveness_check_enabled:
                                is_live, liveness_score, liveness_info = self.liveness_detector.check_liveness(frame, faces[i])
                                # Draw liveness result
                                result_frame = self.liveness_detector.draw_liveness_result(result_frame, faces[i], is_live, liveness_score)
                            
                            # Only proceed with recognition if the face is determined to be live
                            if is_live or not self.liveness_check_enabled:
                                start_time = time.time()
                                person_name, confidence = self.recognize_face(face_image)
                                recognition_time = (time.time() - start_time) * 1000  # ms
                                
                                # Log recognition attempt
                                success = confidence > 0.5
                                self.logger.log_recognition_attempt(
                                    person_name, confidence, recognition_time, success,
                                    lighting="normal", face_angle="frontal")
                                
                                if person_name and confidence > 0.5:
                                    # Add text with recognized name and confidence
                                    label = f"{person_name} ({confidence:.2f})"
                                    cv2.putText(result_frame, label, (x, y-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                # Log failed liveness check
                                self.logger.log_recognition_attempt(
                                    "unknown", 0.0, 0.0, False,
                                    lighting="normal", face_angle="frontal")
            except Exception as e:
                print(f"Error processing faces: {str(e)}")
        
        # Add mode indicator
        cv2.putText(result_frame, f"Mode: {self.current_mode}", 
                    (10, result_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return result_frame
    
    def set_mode(self, mode):
        """
        Set the current operation mode.
        
        Args:
            mode (str): 'detection', 'registration', or 'recognition'
        """
        self.current_mode = mode
        self.mode_label.config(text=f"Mode: {mode.capitalize()}")
    
    def start_registration(self):
        """
        Start the registration process for a new person.
        """
        name = simpledialog.askstring("Registration", "Enter person's name:")
        if name:
            self.current_mode = "registration"
            self.current_person_name = name
            self.current_person_id = self.db.add_person(name)
            self.collected_faces = []
            self.mode_label.config(text="Mode: Registration")
            self.faces_label.config(text="Collected Faces: 0/10")
            messagebox.showinfo("Registration", f"Started registration for {name}. Please look at the camera.")
    
    def collect_face(self, face_image):
        """
        Collect a face image during registration.
        
        Args:
            face_image (numpy.ndarray): Face image to collect
        """
        if self.current_mode == "registration" and self.current_person_id is not None and len(self.collected_faces) < 10:
            self.collected_faces.append(face_image)
    
    def save_faces(self):
        """
        Save collected faces to the database.
        """
        if self.current_mode != "registration" or not self.collected_faces:
            messagebox.showinfo("Save Faces", "No faces collected for registration")
            return
        
        # Save faces to database
        for face_image in self.collected_faces:
            # Generate face encoding
            face_encoding = self.recognizer.extract_features(face_image)
            
            # Add to database
            self.db.add_face(self.current_person_id, face_image, face_encoding)
        
        messagebox.showinfo("Registration Complete", 
                           f"Registration complete for {self.current_person_name} with {len(self.collected_faces)} faces")
        
        # Reset registration state
        self.current_mode = "detection"
        self.mode_label.config(text="Mode: Detection")
        self.current_person_name = ""
        self.current_person_id = None
        self.collected_faces = []
        self.faces_label.config(text="Collected Faces: 0/10")
    
    def train_model(self):
        """
        Train the face recognition model using faces in the database.
        """
        # Get all people from database
        people = self.db.get_all_people()
        if not people:
            messagebox.showinfo("Train Model", "No people in database. Please register people first.")
            return
        
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
            messagebox.showinfo("Train Model", "No face data available for training. Please collect face samples first.")
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Training Model")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)
        
        ttk.Label(progress_window, text="Training model, please wait...").pack(pady=10)
        progress = ttk.Progressbar(progress_window, mode="indeterminate")
        progress.pack(fill=tk.X, padx=20, pady=10)
        progress.start()
        
        # Train in a separate thread to avoid freezing the UI
        def train_thread():
            try:
                # Train the model
                history = self.recognizer.train(face_images, labels)
                
                # Save the model
                os.makedirs("../models", exist_ok=True)
                model_path = "../models/face_recognition_model.h5"
                self.recognizer.save_model(model_path)
                
                # Log training metrics
                self.logger.log_training_metrics(
                    history, "face_recognition_model", 
                    len(set(labels)), len(face_images)
                )
                
                # Close progress window and show success message
                self.root.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", 
                                                             f"Model trained and saved to {model_path}"))
            except Exception as e:
                self.root.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        
        threading.Thread(target=train_thread).start()
    
    def recognize_face(self, face_image):
        """
        Recognize a face in the image.
        
        Args:
            face_image (numpy.ndarray): Face image to recognize
            
        Returns:
            tuple: (person_name, confidence) or (None, 0) if not recognized
        """
        if self.recognizer.model is None:
            return None, 0
        
        return self.recognizer.recognize(face_image)
    
    def update_metrics_display(self):
        """
        Update the performance metrics display.
        """
        metrics = self.logger.get_current_metrics()
        
        self.recognition_rate_label.config(
            text=f"Success Rate: {metrics['success_rate']*100:.1f}%")
        
        self.avg_confidence_label.config(
            text=f"Avg. Confidence: {metrics['avg_confidence']:.2f}")
        
        self.avg_time_label.config(
            text=f"Avg. Time: {metrics['avg_recognition_time']:.1f} ms")
    
    def reset_metrics(self):
        """
        Reset performance metrics.
        """
        self.logger.reset_metrics()
        self.update_metrics_display()
        messagebox.showinfo("Metrics Reset", "Performance metrics have been reset.")
    
    def toggle_liveness_detection(self):
        """
        Toggle liveness detection on/off.
        """
        self.liveness_check_enabled = self.liveness_var.get()
        status = "enabled" if self.liveness_check_enabled else "disabled"
        messagebox.showinfo("Liveness Detection", f"Liveness detection {status}")
        
        # Reset liveness detector when re-enabled
        if self.liveness_check_enabled:
            self.liveness_detector.reset()
    
    def on_closing(self):
        """
        Handle window closing event.
        """
        self.is_running = False
        self.stop_event.set()
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
        if hasattr(self, 'db') and self.db:
            self.db.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FacialRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
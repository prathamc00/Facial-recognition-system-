import os
import csv
import time
import json
import logging
from datetime import datetime

class PerformanceLogger:
    def __init__(self, log_dir="../logs"):
        """
        Initialize the performance logger.
        
        Args:
            log_dir (str): Directory to store log files
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(log_dir, f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize metrics file
        self.metrics_file = os.path.join(log_dir, "recognition_metrics.csv")
        self._initialize_metrics_file()
        
        # Performance metrics
        self.recognition_count = 0
        self.successful_recognitions = 0
        self.confidence_scores = []
        self.recognition_times = []
        
    def _initialize_metrics_file(self):
        """
        Initialize the metrics CSV file with headers if it doesn't exist.
        """
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'person_name', 'confidence', 'recognition_time_ms',
                    'success', 'lighting_condition', 'face_angle'
                ])
    
    def log_recognition_attempt(self, person_name, confidence, recognition_time, 
                               success=True, lighting="normal", face_angle="frontal"):
        """
        Log a face recognition attempt.
        
        Args:
            person_name (str): Name of the recognized person or 'unknown'
            confidence (float): Confidence score (0-1)
            recognition_time (float): Time taken for recognition in milliseconds
            success (bool): Whether recognition was successful
            lighting (str): Lighting condition (e.g., 'bright', 'dim', 'normal')
            face_angle (str): Face angle (e.g., 'frontal', 'side', 'tilted')
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update metrics
        self.recognition_count += 1
        if success:
            self.successful_recognitions += 1
        self.confidence_scores.append(confidence)
        self.recognition_times.append(recognition_time)
        
        # Log to file
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, person_name, confidence, recognition_time,
                success, lighting, face_angle
            ])
        
        # Log to standard logger
        log_message = f"Recognition: {person_name}, Confidence: {confidence:.2f}, Success: {success}"
        logging.info(log_message)
    
    def get_current_metrics(self):
        """
        Get current performance metrics.
        
        Returns:
            dict: Dictionary containing performance metrics
        """
        if self.recognition_count == 0:
            return {
                "total_recognitions": 0,
                "success_rate": 0,
                "avg_confidence": 0,
                "avg_recognition_time": 0
            }
        
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        avg_time = sum(self.recognition_times) / len(self.recognition_times) if self.recognition_times else 0
        
        return {
            "total_recognitions": self.recognition_count,
            "success_rate": self.successful_recognitions / self.recognition_count if self.recognition_count > 0 else 0,
            "avg_confidence": avg_confidence,
            "avg_recognition_time": avg_time
        }
    
    def log_training_metrics(self, history, model_name, num_classes, num_samples):
        """
        Log model training metrics.
        
        Args:
            history: Training history object from Keras
            model_name (str): Name of the model
            num_classes (int): Number of classes (people) in the model
            num_samples (int): Number of training samples used
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert history to dict if it's not already
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        # Create metrics file
        metrics_file = os.path.join(self.log_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Prepare data
        data = {
            "timestamp": timestamp,
            "model_name": model_name,
            "num_classes": num_classes,
            "num_samples": num_samples,
            "metrics": history_dict
        }
        
        # Save to file
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Log summary
        final_acc = history_dict.get('accuracy', [0])[-1]
        final_val_acc = history_dict.get('val_accuracy', [0])[-1]
        logging.info(f"Training completed: {model_name}, Accuracy: {final_acc:.4f}, Validation Accuracy: {final_val_acc:.4f}")
        
    def reset_metrics(self):
        """
        Reset the current metrics counters.
        """
        self.recognition_count = 0
        self.successful_recognitions = 0
        self.confidence_scores = []
        self.recognition_times = []
import cv2
import numpy as np
import time
from scipy.spatial import distance

class LivenessDetector:
    def __init__(self):
        """
        Initialize the liveness detector.
        """
        # Load face landmark detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize dlib's face detector and facial landmark predictor if available
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            # Check if shape_predictor_68_face_landmarks.dat exists, if not, inform user to download it
            import os
            predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
                self.use_dlib = True
            else:
                print(f"Warning: {predictor_path} not found. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("Using basic liveness detection methods instead.")
                self.use_dlib = False
        except ImportError:
            print("dlib not installed. Using basic liveness detection methods.")
            self.use_dlib = False
        
        # Blink detection parameters - fine-tuned for better accuracy
        self.EYE_AR_THRESH = 0.19  # Slightly lower threshold for better sensitivity
        self.EYE_AR_CONSEC_FRAMES = 2  # Reduced for faster detection
        
        # State variables
        self.blink_counter = 0
        self.blink_total = 0
        self.last_blink_time = time.time()
        self.frame_counter = 0
        
        # Motion detection parameters - enhanced
        self.prev_gray = None
        self.motion_threshold = 40  # Lower threshold for better sensitivity
        self.motion_history = []
        self.motion_consistency_threshold = 3  # Number of consistent motion frames required
        
        # Texture analysis parameters
        self.texture_threshold = 400  # Threshold for texture variation
        
        # Adaptive threshold parameters
        self.adaptive_threshold_enabled = True
        self.min_liveness_threshold = 0.55  # Minimum threshold for liveness
        self.max_liveness_threshold = 0.75  # Maximum threshold for liveness
        self.current_liveness_threshold = 0.6  # Default threshold
        self.false_positive_counter = 0
        self.false_negative_counter = 0
        self.threshold_adjustment_rate = 0.01  # Rate at which threshold adjusts
    
    def eye_aspect_ratio(self, eye):
        """
        Calculate the eye aspect ratio (EAR) for blink detection.
        
        Args:
            eye: List of 6 (x, y) coordinates of the eye landmarks
            
        Returns:
            float: Eye aspect ratio
        """
        # Compute the euclidean distances between the vertical eye landmarks
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = distance.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def get_landmarks(self, image, face_rect):
        """
        Get facial landmarks for a detected face.
        
        Args:
            image: Input image
            face_rect: Face rectangle coordinates
            
        Returns:
            numpy.ndarray: Array of landmark points or None if detection fails
        """
        if not self.use_dlib:
            return None
        
        # Convert face_rect from OpenCV format to dlib format
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get facial landmarks
        shape = self.predictor(image, dlib_rect)
        
        # Convert landmarks to numpy array
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
        return landmarks
    
    def detect_eye_reflections(self, frame, landmarks):
        """
        Detect specular highlights (reflections) in the eyes, which are typically
        present in real faces but not in printed photos.
        
        Args:
            frame: Current video frame
            landmarks: Facial landmarks array
            
        Returns:
            float: Score indicating the likelihood of eye reflections being present (0-1)
        """
        if landmarks is None or not self.use_dlib:
            return 0.0
        
        # Extract eye regions
        left_eye_region = self.extract_eye_region(frame, landmarks[42:48])
        right_eye_region = self.extract_eye_region(frame, landmarks[36:42])
        
        reflection_score = 0.0
        
        # Process each eye region
        for eye_region in [left_eye_region, right_eye_region]:
            if eye_region is None or eye_region.size == 0:
                continue
                
            # Convert to grayscale
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to identify bright spots (potential reflections)
            _, thresh = cv2.threshold(gray_eye, 220, 255, cv2.THRESH_BINARY)
            
            # Find contours of bright spots
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (too small or too large are likely noise)
            valid_contours = [c for c in contours if 5 < cv2.contourArea(c) < 100]
            
            # Calculate score based on number and size of reflections
            if len(valid_contours) > 0:
                # More weight for having the right number of reflections (usually 1-2 per eye)
                count_score = min(1.0, len(valid_contours) / 2.0)
                
                # Calculate average brightness of reflection areas
                mask = np.zeros_like(gray_eye)
                cv2.drawContours(mask, valid_contours, -1, 255, -1)
                mean_brightness = cv2.mean(gray_eye, mask=mask)[0] / 255.0
                
                # Combine scores
                eye_score = 0.7 * count_score + 0.3 * mean_brightness
                reflection_score += eye_score / 2.0  # Average across both eyes
        
        return reflection_score
    
    def extract_eye_region(self, frame, eye_landmarks):
        """
        Extract the eye region from the frame using landmarks.
        
        Args:
            frame: Current video frame
            eye_landmarks: Array of 6 (x, y) coordinates for eye landmarks
            
        Returns:
            numpy.ndarray: Extracted eye region or None if extraction fails
        """
        if frame is None or len(eye_landmarks) < 6:
            return None
            
        # Get bounding box of eye
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        
        # Add padding around the eye
        padding = 5
        x_min, x_max = max(0, min(x_coords) - padding), min(frame.shape[1], max(x_coords) + padding)
        y_min, y_max = max(0, min(y_coords) - padding), min(frame.shape[0], max(y_coords) + padding)
        
        # Extract eye region
        eye_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        return eye_region
    
    def detect_blinks(self, landmarks):
        """
        Detect eye blinks using facial landmarks.
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            bool: True if a blink is detected, False otherwise
        """
        if landmarks is None or not self.use_dlib:
            return False
        
        # Extract the left and right eye landmarks
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        
        # Calculate the eye aspect ratios
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the eye aspect ratio is below the blink threshold
        if ear < self.EYE_AR_THRESH:
            self.blink_counter += 1
        else:
            # If the eyes were closed for a sufficient number of frames, count as a blink
            if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.blink_total += 1
                self.last_blink_time = time.time()
                self.blink_counter = 0
                return True
            
            self.blink_counter = 0
        
        return False
    
    def detect_motion(self, frame):
        """
        Detect motion between consecutive frames using an enhanced algorithm.
        
        Args:
            frame: Current video frame
            
        Returns:
            float: Motion magnitude
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)  # Smaller kernel for finer detail
        
        # Initialize previous frame if not set
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0
        
        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        
        # Apply adaptive threshold to highlight regions with significant changes
        # This works better than fixed threshold for varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilate the thresholded image to fill in holes
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to reduce noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        # Calculate total motion as sum of contour areas
        motion = sum([cv2.contourArea(c) for c in valid_contours])
        
        # Calculate motion in facial region specifically if face_rect is provided
        # This helps focus on facial movements rather than background
        
        # Update previous frame with current frame (with a blend to reduce noise)
        alpha = 0.7  # Blend factor
        self.prev_gray = cv2.addWeighted(self.prev_gray, 1-alpha, gray, alpha, 0)
        
        # Add to motion history
        self.motion_history.append(motion)
        if len(self.motion_history) > 15:  # Keep more history for better analysis
            self.motion_history.pop(0)
        
        # Calculate motion consistency (how many frames have consistent motion)
        consistent_motion_frames = sum(1 for m in self.motion_history if m > self.motion_threshold/2)
        
        # Return weighted motion score that considers both magnitude and consistency
        return motion * (1 + 0.2 * min(consistent_motion_frames, self.motion_consistency_threshold))
    
    def analyze_texture(self, face_region):
        """
        Perform advanced texture analysis on the face region for anti-spoofing.
        
        Args:
            face_region: Region of the image containing the face
            
        Returns:
            dict: Dictionary containing texture analysis results
        """
        if face_region.size == 0:
            return {"texture_score": 0, "frequency_score": 0, "gradient_score": 0}
            
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian variance (detail/texture analysis)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        texture_score = min(1.0, laplacian_var / self.texture_threshold)
        
        # 2. Frequency domain analysis (using FFT)
        # Printed/digital faces often have different frequency characteristics
        f_transform = np.fft.fft2(gray_face)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Analyze high frequency components (details present in real faces)
        h, w = gray_face.shape
        center_y, center_x = h//2, w//2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        high_freq_energy = np.mean(magnitude_spectrum[~mask_area])
        frequency_score = min(1.0, high_freq_energy / 1000)
        
        # 3. Gradient analysis (natural faces have smoother gradients)
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_score = min(1.0, np.std(gradient_magnitude) / 50)
        
        return {
            "texture_score": texture_score,
            "frequency_score": frequency_score,
            "gradient_score": gradient_score
        }
    
    def check_liveness(self, frame, face_rect):
        """
        Check if a detected face is from a live person using enhanced methods.
        
        Args:
            frame: Current video frame
            face_rect: Face rectangle coordinates (x, y, w, h)
            
        Returns:
            tuple: (is_live, confidence, liveness_info)
        """
        self.frame_counter += 1
        
        # Extract face region
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        # Initialize liveness score and info
        liveness_score = 0.5  # Default score
        liveness_info = {}
        
        # Check for motion with enhanced detection
        motion = self.detect_motion(frame)
        motion_score = min(1.0, motion / self.motion_threshold)
        liveness_info['motion'] = motion_score
        
        # Get facial landmarks and check for blinks if dlib is available
        if self.use_dlib:
            landmarks = self.get_landmarks(frame, face_rect)
            blink_detected = self.detect_blinks(landmarks)
            
            # Update liveness score based on blink detection
            time_since_last_blink = time.time() - self.last_blink_time
            if blink_detected:
                liveness_score += 0.3
                liveness_info['blink_detected'] = True
            elif time_since_last_blink < 3.0 and self.blink_total > 0:
                liveness_score += 0.2
                liveness_info['recent_blink'] = True
            
            liveness_info['blinks_total'] = self.blink_total
            liveness_info['time_since_blink'] = time_since_last_blink
            
            # Check for eye reflections (specular highlights)
            reflection_score = self.detect_eye_reflections(frame, landmarks)
            liveness_info['reflection_score'] = reflection_score
            
            # Add reflection score contribution to liveness score
            # Eye reflections are strong indicators of a real face
            if reflection_score > 0.5:
                liveness_score += 0.25
            elif reflection_score > 0.3:
                liveness_score += 0.15
        
        # Enhanced texture analysis for anti-spoofing
        if face_region.size > 0:  # Make sure face region is not empty
            texture_results = self.analyze_texture(face_region)
            liveness_info.update(texture_results)
            
            # Weighted contribution from texture analysis
            texture_contribution = (
                texture_results["texture_score"] * 0.4 + 
                texture_results["frequency_score"] * 0.3 + 
                texture_results["gradient_score"] * 0.3
            )
            
            # Add texture contribution to liveness score
            if texture_contribution > 0.5:
                liveness_score += 0.25
        
        # Add motion contribution to liveness score with higher weight
        if motion_score > 0.3:
            liveness_score += 0.15
        
        # Normalize final score to 0-1 range
        liveness_score = max(0.0, min(1.0, liveness_score))
        
        # Use adaptive threshold if enabled
        threshold = self.current_liveness_threshold if self.adaptive_threshold_enabled else 0.6
        is_live = liveness_score > threshold
        
        # Update adaptive threshold based on confidence
        if self.adaptive_threshold_enabled and self.frame_counter % 30 == 0:
            # If score is very high or very low, adjust threshold
            if liveness_score > 0.85 and not is_live:
                # Potential false negative
                self.false_negative_counter += 1
                if self.false_negative_counter > 3:
                    # Lower threshold to reduce false negatives
                    self.current_liveness_threshold = max(
                        self.min_liveness_threshold,
                        self.current_liveness_threshold - self.threshold_adjustment_rate
                    )
                    self.false_negative_counter = 0
            elif liveness_score < 0.4 and is_live:
                # Potential false positive
                self.false_positive_counter += 1
                if self.false_positive_counter > 3:
                    # Raise threshold to reduce false positives
                    self.current_liveness_threshold = min(
                        self.max_liveness_threshold,
                        self.current_liveness_threshold + self.threshold_adjustment_rate
                    )
                    self.false_positive_counter = 0
        
        # Add threshold info to liveness_info
        liveness_info['threshold'] = threshold
        
        return is_live, liveness_score, liveness_info
    
    def draw_liveness_result(self, frame, face_rect, is_live, confidence, liveness_info=None):
        """
        Draw enhanced liveness detection result on the frame with detailed metrics.
        
        Args:
            frame: Video frame to draw on
            face_rect: Face rectangle coordinates (x, y, w, h)
            is_live: Boolean indicating if the face is determined to be live
            confidence: Confidence score for liveness detection
            liveness_info: Optional dictionary with detailed liveness metrics
            
        Returns:
            numpy.ndarray: Frame with liveness result drawn
        """
        x, y, w, h = face_rect
        
        # Create a gradient color based on confidence score
        if is_live:
            # Green gradient (more intense green for higher confidence)
            green_intensity = min(255, int(155 + confidence * 100))
            color = (0, green_intensity, 0)
        else:
            # Red gradient (more intense red for lower confidence)
            red_intensity = min(255, int(155 + (1 - confidence) * 100))
            color = (0, 0, red_intensity)
        
        # Draw rectangle with color based on liveness
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw liveness status text with confidence
        status = "Live" if is_live else "Spoof"
        cv2.putText(frame, f"{status} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw additional metrics if available
        if liveness_info and isinstance(liveness_info, dict):
            # Position for metrics display
            metrics_x = x + w + 10
            metrics_y = y
            line_height = 20
            
            # Display threshold used
            if 'threshold' in liveness_info:
                cv2.putText(frame, f"Threshold: {liveness_info['threshold']:.2f}", 
                            (metrics_x, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 255, 255), 1)
                metrics_y += line_height
            
            # Display key metrics with small indicator bars
            key_metrics = [
                ('motion', 'Motion'), 
                ('texture_score', 'Texture'),
                ('frequency_score', 'Freq'),
                ('gradient_score', 'Gradient'),
                ('reflection_score', 'Eye Refl')
            ]
            
            for key, label in key_metrics:
                if key in liveness_info:
                    # Draw metric name and value
                    value = liveness_info[key]
                    cv2.putText(frame, f"{label}: {value:.2f}", 
                                (metrics_x, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (255, 255, 255), 1)
                    
                    # Draw small indicator bar
                    bar_length = int(value * 50)  # Scale to 50 pixels max
                    bar_height = 5
                    bar_y = metrics_y + 5
                    
                    # Background bar (gray)
                    cv2.rectangle(frame, (metrics_x, bar_y), 
                                 (metrics_x + 50, bar_y + bar_height), 
                                 (100, 100, 100), -1)
                    
                    # Value bar (colored based on value)
                    if value > 0.6:
                        bar_color = (0, 255, 0)  # Green for good values
                    elif value > 0.3:
                        bar_color = (0, 255, 255)  # Yellow for medium values
                    else:
                        bar_color = (0, 0, 255)  # Red for low values
                        
                    cv2.rectangle(frame, (metrics_x, bar_y), 
                                 (metrics_x + bar_length, bar_y + bar_height), 
                                 bar_color, -1)
                    
                    metrics_y += line_height + 10
            
            # Display blink info if available
            if 'blinks_total' in liveness_info:
                blink_text = f"Blinks: {liveness_info['blinks_total']}"
                if 'blink_detected' in liveness_info and liveness_info['blink_detected']:
                    blink_text += " (Detected!)"
                cv2.putText(frame, blink_text, (metrics_x, metrics_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def reset(self):
        """
        Reset the liveness detector state.
        """
        self.blink_counter = 0
        self.blink_total = 0
        self.last_blink_time = time.time()
        self.frame_counter = 0
        self.prev_gray = None
        self.motion_history = []
        self.false_positive_counter = 0
        self.false_negative_counter = 0
        
    def adjust_sensitivity(self, sensitivity_level):
        """
        Adjust the sensitivity of the liveness detection based on the environment.
        
        Args:
            sensitivity_level: Float between 0.0 (least sensitive) and 1.0 (most sensitive)
        """
        # Validate input
        sensitivity_level = max(0.0, min(1.0, sensitivity_level))
        
        # Adjust thresholds based on sensitivity level
        # Higher sensitivity means lower thresholds (more likely to detect as live)
        self.current_liveness_threshold = self.max_liveness_threshold - \
                                         sensitivity_level * (self.max_liveness_threshold - self.min_liveness_threshold)
        
        # Adjust motion threshold (lower = more sensitive)
        self.motion_threshold = 60 - sensitivity_level * 30
        
        # Adjust texture threshold (lower = more sensitive)
        self.texture_threshold = 500 - sensitivity_level * 200
        
        # Adjust eye aspect ratio threshold for blink detection
        self.EYE_AR_THRESH = 0.21 - sensitivity_level * 0.04
        
        # Adjust consecutive frames needed for blink detection
        self.EYE_AR_CONSEC_FRAMES = max(1, int(3 - sensitivity_level * 1.5))
        
        return {
            "liveness_threshold": self.current_liveness_threshold,
            "motion_threshold": self.motion_threshold,
            "texture_threshold": self.texture_threshold,
            "eye_ar_threshold": self.EYE_AR_THRESH,
            "eye_ar_consec_frames": self.EYE_AR_CONSEC_FRAMES
        }
        
    def reset_adaptive_threshold(self):
        """
        Reset the adaptive threshold to its default value.
        """
        self.current_liveness_threshold = 0.6
        self.false_positive_counter = 0
        self.false_negative_counter = 0
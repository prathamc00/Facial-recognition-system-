import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

class RaspberryPiFaceDetector:
    def __init__(self, cascade_path=None, resolution=(640, 480), fps=3):
        """
        Initialize the Raspberry Pi face detector with picamera2 support.
        
        Args:
            cascade_path (str, optional): Path to the Haar cascade XML file.
            resolution (tuple): Camera resolution (width, height)
            fps (int): Target frames per second for processing
        """
        if cascade_path is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade classifier. Check the path.")
        
        # Initialize picamera2
        self.camera = Picamera2()
        self.configure_camera(resolution, fps)
    
    def configure_camera(self, resolution, fps):
        """
        Configure the camera with optimized settings for face detection.
        """
        # Create a camera configuration for the main stream
        config = self.camera.create_still_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={
                "FrameDurationLimits": (int(1/fps * 1000000), int(1/fps * 1000000)),
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast
            }
        )
        self.camera.configure(config)
    
    def start_camera(self):
        """Start the camera stream."""
        self.camera.start()
    
    def stop_camera(self):
        """Stop the camera stream."""
        self.camera.stop()
    
    def get_frame(self):
        """Capture a frame from the camera."""
        return self.camera.capture_array()
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image with optimized parameters for Raspberry Pi.
        
        Args:
            image: Input image (numpy array)
            scale_factor: Parameter specifying how much the image size is reduced
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size
            
        Returns:
            List of tuples containing face coordinates (x, y, w, h)
        """
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces.
        
        Args:
            image: Input image
            faces: List of face coordinates (x, y, w, h)
            color: Rectangle color (R, G, B)
            thickness: Line thickness
            
        Returns:
            Image with rectangles drawn around faces
        """
        img_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, thickness)
        return img_copy
    
    def extract_faces(self, image, faces, target_size=(224, 224)):
        """
        Extract and resize face regions from the image.
        
        Args:
            image: Input image
            faces: List of face coordinates (x, y, w, h)
            target_size: Size to resize the extracted faces to
            
        Returns:
            List of extracted and resized face images
        """
        face_images = []
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, target_size)
            face_images.append(face_resized)
        return face_images

# Example usage
if __name__ == "__main__":
    # Create a face detector
    detector = RaspberryPiFaceDetector(resolution=(640, 480), fps=3)
    
    try:
        # Start the camera
        detector.start_camera()
        
        while True:
            # Capture frame
            frame = detector.get_frame()
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw rectangles around faces
            result_frame = detector.draw_faces(frame, faces)
            
            # Display the result
            cv2.imshow("Detected Faces", result_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        detector.stop_camera()
        cv2.destroyAllWindows()
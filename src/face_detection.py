import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, cascade_path=None):
        """
        Initialize the face detector with a Haar cascade classifier.
        
        Args:
            cascade_path (str, optional): Path to the Haar cascade XML file.
                If None, uses the default OpenCV cascade.
        """
        if cascade_path is None:
            # Use the default OpenCV Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade classifier. Check the path.")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image.
        
        Args:
            image: Input image (numpy array)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size. Objects smaller than this are ignored
            
        Returns:
            List of tuples containing face coordinates (x, y, w, h)
        """
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces.
        
        Args:
            image: Input image
            faces: List of face coordinates (x, y, w, h)
            color: Rectangle color (B, G, R)
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
        Extract face regions from the image and resize them.
        
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
            # Resize to target size
            face_resized = cv2.resize(face, target_size)
            face_images.append(face_resized)
        return face_images

# Example usage
if __name__ == "__main__":
    # Create a face detector
    detector = FaceDetector()
    
    # Load an image
    image_path = "../data/test_image.jpg"  # Update with your image path
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        print(f"Found {len(faces)} faces!")
        
        # Draw rectangles around faces
        result_image = detector.draw_faces(image, faces)
        
        # Display the result
        cv2.imshow("Detected Faces", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image not found at {image_path}")
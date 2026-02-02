import os
import cv2
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from face_detection import FaceDetector

class FaceRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the face recognizer.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file.
                If None, a new model will be created but not trained.
        """
        self.model = None
        self.label_encoder = LabelEncoder()
        self.face_detector = FaceDetector()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self, num_classes, input_shape=(224, 224, 3)):
        """
        Build a face recognition model using transfer learning with MobileNetV2.
        
        Args:
            num_classes: Number of people to recognize
            input_shape: Input image dimensions
            
        Returns:
            Compiled Keras model
        """
        # Use MobileNetV2 as the base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, face_images, labels, validation_split=0.2, epochs=20, batch_size=32, 
              use_augmentation=False, augmentation_factor=3):
        """
        Train the face recognition model with enhanced callbacks and optional augmentation.
        
        Args:
            face_images: List or array of face images
            labels: List or array of corresponding labels (person names/IDs)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            use_augmentation: Whether to use data augmentation
            augmentation_factor: Number of augmented versions per image if using augmentation
            
        Returns:
            Training history
        """
        # Apply data augmentation if requested
        if use_augmentation:
            try:
                from data_augmentation import DataAugmentation
                aug = DataAugmentation()
                face_images, labels = aug.create_training_set(
                    face_images, labels, augmentation_factor=augmentation_factor
                )
                print(f"Data augmentation applied: {len(face_images)} total images")
            except ImportError:
                print("Warning: data_augmentation module not found. Skipping augmentation.")
        
        # Convert labels to numerical encoding
        self.label_encoder.fit(labels)
        numerical_labels = self.label_encoder.transform(labels)
        
        # Convert to one-hot encoding
        categorical_labels = tf.keras.utils.to_categorical(numerical_labels)
        
        # Build model if it doesn't exist
        if self.model is None:
            self.build_model(len(self.label_encoder.classes_))
        
        # Preprocess images
        processed_images = np.array([self._preprocess_image(img) for img in face_images])
        
        # Create a new optimizer instance for each training session
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks for better training
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            # Stop training when validation loss doesn't improve for 5 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            processed_images, categorical_labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    
    def recognize(self, face_image):
        """
        Recognize a face and return the predicted label and confidence.
        
        Args:
            face_image: Image containing a face
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Preprocess the image
        processed_image = self._preprocess_image(face_image)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = self.model.predict(processed_image)[0]
        
        # Get the highest confidence prediction
        max_index = np.argmax(predictions)
        confidence = predictions[max_index]
        
        # Convert numerical label back to original label
        predicted_label = self.label_encoder.inverse_transform([max_index])[0]
        
        return predicted_label, confidence
    
    def recognize_from_image(self, image):
        """
        Detect and recognize all faces in an image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of tuples containing (face_coordinates, predicted_label, confidence)
        """
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        # Extract face images
        face_images = self.face_detector.extract_faces(image, faces)
        
        results = []
        for i, face_img in enumerate(face_images):
            # Recognize each face
            label, confidence = self.recognize(face_img)
            results.append((faces[i], label, confidence))
        
        return results
    
    def save_model(self, model_path):
        """
        Save the trained model and label encoder.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Save the model
        self.model.save(model_path)
        
        # Save the label encoder classes
        np.save(os.path.splitext(model_path)[0] + '_labels.npy', self.label_encoder.classes_)
    
    def load_model(self, model_path):
        """
        Load a trained model and label encoder.
        
        Args:
            model_path: Path to the saved model
        """
        # Load the model
        self.model = load_model(model_path)
        
        # Load the label encoder classes
        labels_path = os.path.splitext(model_path)[0] + '_labels.npy'
        if os.path.exists(labels_path):
            self.label_encoder.classes_ = np.load(labels_path, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Label encoder data not found at {labels_path}")
    
    def extract_features(self, face_image):
        """
        Extract features from a face image for recognition or storage.
        
        Args:
            face_image: Face image to extract features from
            
        Returns:
            numpy.ndarray: Feature vector representing the face
        """
        # Preprocess the image
        processed_image = self._preprocess_image(face_image)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
        if self.model is None:
            # If no model is loaded, return a random feature vector (placeholder)
            # In a real application, you might want to use a dedicated feature extractor
            return np.random.rand(128)  # 128-dimensional feature vector
        
        # Use the model's intermediate layer as a feature extractor
        # Get the output of the layer before the final classification layer
        feature_model = Model(inputs=self.model.input, 
                             outputs=self.model.layers[-3].output)  # -3 to skip final Dense and Dropout
        
        # Extract features
        features = feature_model.predict(processed_image)[0]
        return features
    
    def _preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess an image for the model.
        
        Args:
            image: Input image
            target_size: Target size for the image
            
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            image = cv2.resize(image, target_size)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize
        image = img_to_array(image) / 255.0
        
        return image

# Example usage
if __name__ == "__main__":
    # Create a face recognizer
    recognizer = FaceRecognizer()
    
    # Example of how to train (you would need actual face data)
    # recognizer.build_model(num_classes=5)  # For 5 different people
    # history = recognizer.train(face_images, labels, epochs=10)
    # recognizer.save_model("../models/face_recognition_model.h5")
    
    # Example of how to use for recognition
    # Load an image
    image_path = "../data/test_image.jpg"  # Update with your image path
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        
        # Detect and recognize faces
        results = recognizer.recognize_from_image(image)
        
        # Draw results on the image
        for (x, y, w, h), label, confidence in results:
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("Face Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image not found at {image_path}")
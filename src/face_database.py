import os
import sqlite3
import numpy as np
import cv2

class FaceDatabase:
    def __init__(self, db_path):
        """
        Initialize the face database.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.connect()
        self._create_tables()
    
    def connect(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        # People table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        
        # Face data table with quality metrics and confidence scores
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                face_image BLOB NOT NULL,
                face_encoding BLOB,
                quality_score FLOAT,
                brightness_score FLOAT,
                sharpness_score FLOAT,
                confidence_score FLOAT,
                recognition_count INTEGER DEFAULT 0,
                last_recognition_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people (id)
            )""")
        
        self.conn.commit()
    
    def add_person(self, name):
        """
        Add a new person to the database.
        
        Args:
            name (str): Name of the person
            
        Returns:
            int: ID of the newly added person
        """
        self.cursor.execute("INSERT INTO people (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_face(self, person_id, face_image, face_encoding, quality_metrics=None):
        """
        Add a face image and its encoding to the database with quality metrics.
        
        Args:
            person_id (int): ID of the person this face belongs to
            face_image (numpy.ndarray): Face image as numpy array
            face_encoding (numpy.ndarray): Face encoding as numpy array
            quality_metrics (dict, optional): Dictionary containing quality metrics
                                            (quality_score, brightness_score, sharpness_score)
        
        Raises:
            ValueError: If face_image or face_encoding is invalid, or if person_id doesn't exist
        """
        # Validate inputs
        if not isinstance(person_id, int) or person_id <= 0:
            raise ValueError("Invalid person_id: must be a positive integer")
        
        if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
            raise ValueError("Invalid face_image: must be a non-empty numpy array")
            
        if face_encoding is None or not isinstance(face_encoding, np.ndarray) or face_encoding.size == 0:
            raise ValueError("Invalid face_encoding: must be a non-empty numpy array")
            
        # Verify person exists
        self.cursor.execute("SELECT id FROM people WHERE id = ?", (person_id,))
        if not self.cursor.fetchone():
            raise ValueError(f"Person with id {person_id} does not exist")
            
        try:
            # Convert numpy arrays to binary data for storage
            _, img_buffer = cv2.imencode('.jpg', face_image)
            img_bytes = img_buffer.tobytes()
            encoding_bytes = face_encoding.tobytes()
        except Exception as e:
            raise ValueError(f"Failed to encode face data: {str(e)}")
        
        # Default quality metrics if not provided
        if quality_metrics is None:
            quality_metrics = {
                'quality_score': self._calculate_quality_score(face_image),
                'brightness_score': self._calculate_brightness(face_image),
                'sharpness_score': self._calculate_sharpness(face_image)
            }
        
        self.cursor.execute(
            """INSERT INTO face_data 
            (person_id, face_image, face_encoding, quality_score, brightness_score, 
             sharpness_score, confidence_score) 
            VALUES (?, ?, ?, ?, ?, ?, 0.0)""",
            (person_id, img_bytes, encoding_bytes, 
             quality_metrics['quality_score'],
             quality_metrics['brightness_score'],
             quality_metrics['sharpness_score'])
        )
        self.conn.commit()

    def _calculate_quality_score(self, face_image):
        """Calculate overall quality score for face image using multiple metrics.
        
        Args:
            face_image (numpy.ndarray): Face image as numpy array
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
            return 0.0
            
        try:
            # Calculate individual metrics
            brightness = self._calculate_brightness(face_image)
            sharpness = self._calculate_sharpness(face_image)
            
            # Check face size (assuming minimum face size should be 96x96)
            size_score = min(1.0, face_image.shape[0] / 96.0 * face_image.shape[1] / 96.0)
            
            # Combine metrics with weights
            quality_score = 0.4 * brightness + 0.4 * sharpness + 0.2 * size_score
            
            return min(1.0, max(0.0, quality_score))
        except Exception:
            return 0.0

    def _calculate_brightness(self, face_image):
        """Calculate brightness score for face image.
        
        Args:
            face_image (numpy.ndarray): Face image as numpy array
            
        Returns:
            float: Brightness score between 0.0 and 1.0
        """
        if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
            return 0.0
            
        try:
            # Convert to grayscale and calculate mean brightness
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray) / 255.0
            
            # Penalize too dark or too bright images
            if mean_brightness < 0.2 or mean_brightness > 0.8:
                return 0.5 * mean_brightness
                
            return mean_brightness
        except Exception:
            return 0.0

    def _calculate_sharpness(self, face_image):
        """Calculate sharpness score for face image using Laplacian variance.
        
        Args:
            face_image (numpy.ndarray): Face image as numpy array
            
        Returns:
            float: Sharpness score between 0.0 and 1.0
        """
        if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
            return 0.0
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance as sharpness measure
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize with a more appropriate scale factor
            # Most good quality images have variance between 100 and 1000
            sharpness_score = laplacian_var / 500.0
            
            return min(1.0, max(0.0, sharpness_score))
        except Exception:
            return 0.0
    
    def get_all_people(self):
        """
        Get all people from the database.
        
        Returns:
            list: List of dictionaries containing person information
        """
        self.cursor.execute("SELECT id, name FROM people")
        return [{'id': row[0], 'name': row[1]} for row in self.cursor.fetchall()]
    
    def get_person_faces(self, person_id):
        """
        Get all face data for a specific person.
        
        Args:
            person_id (int): ID of the person
            
        Returns:
            list: List of dictionaries containing face data
            
        Raises:
            ValueError: If person_id is invalid or person does not exist
        """
        if not isinstance(person_id, int) or person_id <= 0:
            raise ValueError("Invalid person_id: must be a positive integer")
            
        # Verify person exists
        self.cursor.execute("SELECT id FROM people WHERE id = ?", (person_id,))
        if not self.cursor.fetchone():
            raise ValueError(f"Person with id {person_id} does not exist")
            
        self.cursor.execute("SELECT id, face_image, face_encoding FROM face_data WHERE person_id = ?", (person_id,))
        
        faces = []
        for row in self.cursor.fetchall():
            try:
                # Convert binary data back to numpy arrays
                img_array = np.frombuffer(row[1], dtype=np.uint8)
                face_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if face_image is None:
                    continue  # Skip corrupted images
                
                face_encoding = np.frombuffer(row[2], dtype=np.float32)
                
                faces.append({
                    'id': row[0],
                    'face_image': face_image,
                    'face_encoding': face_encoding
                })
            except Exception as e:
                print(f"Warning: Failed to decode face data for ID {row[0]}: {str(e)}")
                continue
        
        return faces
    
    def update_recognition_metrics(self, face_id, confidence_score):
        """
        Update recognition metrics for a face after successful recognition.
        
        Args:
            face_id (int): ID of the recognized face
            confidence_score (float): Confidence score of the recognition
        """
        self.cursor.execute("""
            UPDATE face_data 
            SET recognition_count = recognition_count + 1,
                confidence_score = ?,
                last_recognition_time = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (confidence_score, face_id))
        self.conn.commit()
    
    def get_face_quality_info(self, face_id):
        """
        Get quality metrics and recognition statistics for a face.
        
        Args:
            face_id (int): ID of the face
            
        Returns:
            dict: Dictionary containing quality metrics and recognition statistics
        """
        self.cursor.execute("""
            SELECT quality_score, brightness_score, sharpness_score,
                   confidence_score, recognition_count, last_recognition_time
            FROM face_data
            WHERE id = ?
        """, (face_id,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'quality_score': row[0],
                'brightness_score': row[1],
                'sharpness_score': row[2],
                'confidence_score': row[3],
                'recognition_count': row[4],
                'last_recognition_time': row[5]
            }
        return None

    def get_face_encodings(self, person_id=None, min_quality_score=None):
        """
        Get face encodings for all or specific person with quality filtering.
        
        Args:
            person_id (int, optional): If specified, only get encodings for this person
            min_quality_score (float, optional): Minimum quality score threshold
            
        Returns:
            tuple: (encodings, labels) where encodings is a list of numpy arrays
                   and labels is a list of corresponding person names
                   
        Raises:
            ValueError: If person_id is invalid or person does not exist
        """
        if person_id is not None:
            if not isinstance(person_id, int) or person_id <= 0:
                raise ValueError("Invalid person_id: must be a positive integer")
                
            # Verify person exists
            self.cursor.execute("SELECT id FROM people WHERE id = ?", (person_id,))
            if not self.cursor.fetchone():
                raise ValueError(f"Person with id {person_id} does not exist")
        
        if min_quality_score is not None:
            if not isinstance(min_quality_score, (int, float)) or not 0 <= min_quality_score <= 1:
                raise ValueError("Invalid min_quality_score: must be a float between 0 and 1")
        
        query = """
            SELECT p.name, f.face_encoding, f.id 
            FROM face_data f 
            JOIN people p ON f.person_id = p.id 
            WHERE f.face_encoding IS NOT NULL
        """
        params = []
        
        if person_id:
            query += " AND f.person_id = ?"
            params.append(person_id)
        
        if min_quality_score is not None:
            query += " AND f.quality_score >= ?"
            params.append(min_quality_score)
        
        self.cursor.execute(query, params)
        
        encodings = []
        labels = []
        
        for row in self.cursor.fetchall():
            try:
                encoding = np.frombuffer(row[1], dtype=np.float32)
                if encoding.size > 0:  # Skip empty encodings
                    labels.append(row[0])
                    encodings.append(encoding)
            except Exception as e:
                print(f"Warning: Failed to decode face encoding for ID {row[2]}: {str(e)}")
                continue
        
        return encodings, labels
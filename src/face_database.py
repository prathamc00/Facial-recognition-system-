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
        
        # Faces table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            face_image BLOB,
            face_encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(person_id) REFERENCES people(id)
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
    
    def add_face(self, person_id, face_image, face_encoding):
        """
        Add a face image and its encoding to the database.
        
        Args:
            person_id (int): ID of the person this face belongs to
            face_image (numpy.ndarray): Face image as numpy array
            face_encoding (numpy.ndarray): Face encoding as numpy array
        """
        # Convert numpy arrays to binary data for storage
        _, img_buffer = cv2.imencode('.jpg', face_image)
        img_bytes = img_buffer.tobytes()
        encoding_bytes = face_encoding.tobytes()
        
        self.cursor.execute(
            "INSERT INTO faces (person_id, face_image, face_encoding) VALUES (?, ?, ?)",
            (person_id, img_bytes, encoding_bytes)
        )
        self.conn.commit()
    
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
        """
        self.cursor.execute("SELECT id, face_image, face_encoding FROM faces WHERE person_id = ?", (person_id,))
        
        faces = []
        for row in self.cursor.fetchall():
            # Convert binary data back to numpy arrays
            img_array = np.frombuffer(row[1], dtype=np.uint8)
            face_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            face_encoding = np.frombuffer(row[2], dtype=np.float32)
            
            faces.append({
                'id': row[0],
                'face_image': face_image,
                'face_encoding': face_encoding
            })
        
        return faces
    
    def get_face_encodings(self, person_id=None):
        """
        Get face encodings for all or specific person.
        
        Args:
            person_id (int, optional): If specified, only get encodings for this person
            
        Returns:
            tuple: (encodings, labels) where encodings is a list of numpy arrays
                   and labels is a list of corresponding person names
        """
        if person_id:
            query = """
                SELECT p.name, f.face_encoding 
                FROM faces f 
                JOIN people p ON f.person_id = p.id 
                WHERE f.person_id = ?
            """
            params = (person_id,)
        else:
            query = """
                SELECT p.name, f.face_encoding 
                FROM faces f 
                JOIN people p ON f.person_id = p.id
            """
            params = ()
        
        self.cursor.execute(query, params)
        
        encodings = []
        labels = []
        
        for row in self.cursor.fetchall():
            labels.append(row[0])
            encodings.append(np.frombuffer(row[1], dtype=np.float32))
        
        return encodings, labels
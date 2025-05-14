import numpy as np
import dlib
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    """Class to recognize faces based on embeddings"""
    
    def __init__(self, model_path="models/dlib_face_recognition_resnet_model_v1.dat",
                database_path="data/face_database.pkl"):
        """
        Initialize the face recognizer
        
        Args:
            model_path: Path to the face embedding model
            database_path: Path to the face database
        """
        # Load the face embedding model
        self.face_encoder = dlib.face_recognition_model_v1(model_path)
        
        # Face database
        self.database_path = database_path
        self.face_database = {}
        self.load_database()
        
        # Similarity threshold (adjust as needed)
        self.similarity_threshold = 0.6
    
    def load_database(self):
        """Load the face database from disk"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"Face database loaded with {len(self.face_database)} people")
            except Exception as e:
                print(f"Error loading face database: {e}")
                self.face_database = {}
    
    def save_database(self):
        """Save the face database to disk"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.face_database, f)
        print(f"Face database saved with {len(self.face_database)} people")
    
    def get_face_embedding(self, frame, landmarks):
        """
        Extract the face embedding (vector of features)
        
        Args:
            frame: Image/frame
            landmarks: Facial landmarks
        
        Returns:
            Face embedding (128 dimensions)
        """
        # Convert landmarks to dlib format
        dlib_shape = dlib.full_object_detection(
            dlib.rectangle(0, 0, frame.shape[1], frame.shape[0]),
            [dlib.point(x, y) for x, y in landmarks]
        )
        
        # Extract embedding
        face_descriptor = self.face_encoder.compute_face_descriptor(
            frame, dlib_shape, 1)
        
        return np.array(face_descriptor)
    
    def register_face(self, person_id, embedding, metadata=None):
        """
        Register a face in the database
        
        Args:
            person_id: Unique identifier of the person
            embedding: Face embedding
            metadata: Additional information (name, position, etc.)
        """
        if person_id not in self.face_database:
            self.face_database[person_id] = {
                'embeddings': [],
                'metadata': metadata or {}
            }
        
        self.face_database[person_id]['embeddings'].append(embedding)
        self.save_database()
    
    def identify_face(self, embedding):
        """
        Identify a face based on the embedding
        
        Args:
            embedding: Face embedding
        
        Returns:
            (person_id, similarity, metadata) or (None, 0, None) if not recognized
        """
        if not self.face_database:
            return None, 0, None
        
        best_match = None
        best_similarity = -1
        
        for person_id, data in self.face_database.items():
            for stored_embedding in data['embeddings']:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [embedding], [stored_embedding])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id
        
        # Check if the similarity is above the threshold
        if best_similarity >= self.similarity_threshold:
            return (best_match, 
                    best_similarity, 
                    self.face_database[best_match]['metadata'])
        else:
            return None, best_similarity, None

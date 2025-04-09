import cv2
import os
import numpy as np
from ..config.settings import KNOWN_FACES_DIR, FACE_MATCH_THRESHOLD

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_faces = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load and process known faces"""
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            return
        
        print("Loading known faces...")
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                try:
                    image_path = os.path.join(KNOWN_FACES_DIR, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Could not load image: {filename}")
                        continue
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    if len(faces) > 0:
                        # Get the largest face
                        face = max(faces, key=lambda x: x[2] * x[3])
                        x, y, w, h = face
                        
                        # Extract face region and features
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (64, 64))
                        
                        # Calculate features
                        features = self.extract_face_features(face_roi)
                        
                        name = os.path.splitext(filename)[0]
                        self.known_faces.append({
                            'name': name,
                            'features': features,
                            'face_size': (w, h)
                        })
                        print(f"Loaded known face: {name}")
                
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        print(f"Loaded {len(self.known_faces)} known faces")
    
    def extract_face_features(self, face_img):
        """Extract features from face image using LBP"""
        features = []
        
        # Calculate LBP features
        for y in range(1, face_img.shape[0] - 1):
            for x in range(1, face_img.shape[1] - 1):
                center = face_img[y, x]
                binary = (face_img[y-1:y+2, x-1:x+2] >= center).flatten()
                binary = binary.astype(int)
                binary = np.delete(binary, 4)  # Remove center
                decimal = sum([val * (2**idx) for idx, val in enumerate(binary)])
                features.append(decimal)
        
        features = np.array(features, dtype=np.float32)
        if np.any(features):
            features = features / np.linalg.norm(features)
        
        return features
    
    def compare_faces(self, features1, features2):
        """Compare two face feature vectors"""
        if len(features1) != len(features2):
            return 0
        return np.dot(features1, features2)
    
    def detect_faces(self, frame, draw=True):
        """Detect and recognize faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        recognized_faces = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            features = self.extract_face_features(face_roi)
            
            # Compare with known faces
            max_similarity = 0
            recognized_name = "Unknown"
            
            for known_face in self.known_faces:
                similarity = self.compare_faces(features, known_face['features'])
                if similarity > max_similarity and similarity > FACE_MATCH_THRESHOLD:
                    max_similarity = similarity
                    recognized_name = known_face['name']
            
            recognized_faces.append({
                'name': recognized_name,
                'box': (x, y, w, h),
                'similarity': max_similarity
            })
            
            if draw:
                # Draw rectangle and name
                color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, recognized_name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, recognized_faces 
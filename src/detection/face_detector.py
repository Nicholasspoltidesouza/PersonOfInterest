import cv2
import numpy as np  
import dlib

class FaceDetector:
    """Class to detect faces in a video stream"""
    def __init__(self, detection_method="hog"):
        """Initialize the face detector
        
        Args:
            detection_method (str): The method to use for face detection (hog or cnn)
        """
        self.detection_method = detection_method
        self.hog_detector = dlib.get_frontal_face_detector()
        self.cnn_detector = None
        if detection_method == "cnn":
            try:
                self.cnn_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
            except Exception as e:
                print(f"Error loading CNN face detector: {e}")
                self.detection_method = "hog"

        self.landmarkd_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def detect_faces(self, frame):
        """Detect faces in a frame
        
        Args:
            frame (numpy.ndarray): Image/frame to detect faces in

        Returns:
            list: A list of rectangles containing the detected faces (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.detection_method == "hog":
            dlib_rects = self.hog_detector(gray, 1)
            face_rects = []
            for rect in dlib_rects:
                x = rect.left()
                y = rect.top()
                w = rect.right() - rect.left()
                h = rect.bottom() - rect.top()
                face_rects.append((x, y, w, h))
        else:
            dlib_mmod_rects = self.cnn_detector(gray, 1)
            face_rects = []
            for rect in dlib_mmod_rects:
                x = rect.rect.left()
                y = rect.rect.top()
                w = rect.rect.right() - rect.rect.left()
                h = rect.rect.bottom() - rect.rect.top()
                face_rects.append((x, y, w, h))
        
        return face_rects
    
    def get_landmarks(self, frame, face_rect):
        """Get the landmarks of a face
        
        Args:
            frame (numpy.ndarray): Image/frame to get landmarks in
            face_rect (tuple): Rectangle containing the face (x, y, w, h)

        Returns:
            Array of shape (68, 2) containing the landmarks of the face
        """
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.landmark_predictor(gray, dlib_rect)

        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks
    
    def align_face(self, frame, landmarks):
        """Align a face based on the landmarks for better recognition

        Args:
            frame (numpy.ndarray): Image/frame to align face in
            landmarks (numpy.ndarray): Array of shape (68, 2) containing the landmarks of the face

        Returns:
            Aligned face
        """
        left_eye = landmarks[36:42].mean(axis=0).astype("int")
        right_eye = landmarks[42:48].mean(axis=0).astype("int")

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        height, width = frame.shape[:2]
        aligned_face = cv2.warpAffine(frame, M, (width, height), flags=cv2.INTER_CUBIC)

        return aligned_face

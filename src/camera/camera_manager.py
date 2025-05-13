import cv2
import time
from threading import Thread

class VideoStream:
    """Class to manage videos streams from different sources"""
    def __init__(self, src=0 , name="VideoStream", width=640, height=480):
        """
        Initialize the video stream

        Args:
            src (int): The source of the video stream (0 for webcam, string for IP/RTSP)
            name (str): The name of the video stream
            width (int): The width of the video stream
            height (int): The height of the video stream
        """
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.name = name
        self.stopped = False
        self.frame = None
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0

    def start(self):
        """Start the thread to read frames from the video stream"""
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        """Primal loop to read frames from the video stream"""
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break
            
            self.frame = frame
            
            self.frame_count += 1
            elapsed_time = time.time() - self.last_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.last_time = time.time()

    def read(self):
        """Return the next frame from the stream"""
        return self.frame
    
    def get_fps(self):
        """Return the current fps of the video stream"""
        return self.fps
    
    def stop(self):
        """Stop the thread and release the video stream"""
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

class CameraManager:
    """Class to manage multiple video streams"""

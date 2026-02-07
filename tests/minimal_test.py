import sys
import os
import numpy as np
import cv2

class MinimalKF:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 2)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

    def initialize(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0], [0], [0.5]], dtype=np.float32)
        
    def predict(self):
        return self.kf.predict()[:2]
        
    def update(self, x, y):
        m = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(m)

class MinimalTracker:
    def __init__(self):
        self.kf = MinimalKF()
        self.state = "BOOTSTRAP"
        self.count = 0

    def process_frame(self, frame, i, detector):
        res = detector.detect(frame)
        if res:
            x, y = res[0][:2]
            if self.count == 0:
                self.kf.initialize(x, y)
            else:
                self.kf.predict()
                self.kf.update(x, y)
            self.count += 1
            return {'box': [x, y, 10, 10], 'source': 'mock'}
        return None

class Mock:
    def detect(self, f):
        return [[100, 100, 10, 10]]

def test():
    tracker = MinimalTracker()
    mock = Mock()
    for i in range(10):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        print(f"Step {i}")
        info = tracker.process_frame(frame, i, mock)
        print(f"Result: {info}")

if __name__ == "__main__":
    test()
    print("SUCCESS")

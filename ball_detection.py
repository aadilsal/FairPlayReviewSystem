import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import LinearSVC
# from ultralytics import YOLO  # Temporarily commented out due to import issues


class BallDetector:
    def __init__(self, window_size=(64, 64), step_size=16):
        self.window_size = window_size
        self.step_size = step_size
        self.svm = None
        self.yolo = None  # YOLOv8 model

    # ---------- SLIDING WINDOW ----------
    def extract_hog_features(self, image):
        return hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False, channel_axis=-1)

    def train(self, positive_images, negative_images):
        """Train SVM using positive (ball) and negative (non-ball) samples"""
        X, y = [], []

        for img in positive_images:
            img = cv2.resize(img, self.window_size)
            X.append(self.extract_hog_features(img))
            y.append(1)

        for img in negative_images:
            img = cv2.resize(img, self.window_size)
            X.append(self.extract_hog_features(img))
            y.append(0)

        self.svm = LinearSVC(max_iter=5000)
        self.svm.fit(X, y)

    def sliding_window_detect(self, image):
        detections = []
        for y in range(0, image.shape[0] - self.window_size[1], self.step_size):
            for x in range(0, image.shape[1] - self.window_size[0], self.step_size):
                window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                if window.shape[:2] != self.window_size:
                    continue
                feat = self.extract_hog_features(window)
                if self.svm is not None and self.svm.predict([feat])[0] == 1:
                    detections.append((x, y, self.window_size[0], self.window_size[1]))
        return detections

    # ---------- COLOR SEGMENTATION ----------
    def color_segmentation(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # cricket balls are red/orange
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((x, y, w, h))
        return detections

    # ---------- YOLOv8 ----------
    def load_yolo(self, dataset_path=None):
        """Load YOLOv8 model, fine-tune on dataset if provided"""
        print("YOLO functionality temporarily disabled due to import issues")
        print("Using only SVM and color segmentation for now")
        # if dataset_path and os.path.exists(os.path.join(dataset_path, "data.yaml")):
        #     # fine-tune on cricket ball dataset
        #     print(f"Training YOLO model on local dataset: {dataset_path}")
        #     self.yolo = YOLO("yolov8n.pt")
        #     try:
        #         self.yolo.train(data=os.path.join(dataset_path, "data.yaml"), epochs=10, imgsz=640)
        #         print("YOLO training completed successfully")
        #     except Exception as e:
        #         print(f"YOLO training failed: {e}")
        #         print("Falling back to pretrained model")
        #         self.yolo = YOLO("yolov8n.pt")
        # else:
        #     # fallback to pretrained small model
        #     print("Using pretrained YOLO model")
        #     self.yolo = YOLO("yolov8n.pt")

    def yolo_detect(self, image):
        detections = []
        if self.yolo is None:
            return detections
        # results = self.yolo.predict(source=image, verbose=False)
        # for r in results:
        #     for box in r.boxes.xyxy.cpu().numpy():
        #         x1, y1, x2, y2 = box[:4].astype(int)
        #         detections.append((x1, y1, x2-x1, y2-y1))
        return detections

    # ---------- FINAL DETECTION PIPELINE ----------
    def detect(self, image):
        detections = []

        # Sliding Window
        if self.svm:
            detections.extend(self.sliding_window_detect(image))

        # Color Segmentation
        detections.extend(self.color_segmentation(image))

        # YOLOv8 (temporarily disabled)
        # if self.yolo:
        #     detections.extend(self.yolo_detect(image))

        return detections

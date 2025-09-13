import cv2
from ultralytics import YOLO

class YOLOBallDetector:
    def __init__(self, model_path='outputs/yolov8_cricket_ball2/weights/best.pt'):
        self.model = YOLO(model_path)

    def detect(self, img):
        detections = []
        results = self.model(img)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # For your custom model, cricket ball is class 0
                    if int(box.cls) == 0:
                        confidence = float(box.conf)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                        detections.append((x, y, w, h, confidence))
        return detections

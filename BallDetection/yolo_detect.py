import os
import logging
import numpy as np
import torch
from ultralytics import YOLO
from global_config import GLOBAL_CONFIG
from BallDetection.config import DETECTION_CONFIG

logger = logging.getLogger(__name__)

class YOLOBallDetector:
    def __init__(self, model_path='weights/ball-yolov8s.pt', device=None):
        self.model = None
        self.model_path = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_weights(model_path)

    def load_weights(self, model_path):
        load_path = model_path if model_path and os.path.isfile(model_path) else 'weights/ball-yolov8s.pt'  
        try:
            self.model = YOLO(load_path)
            self.model_path = load_path
            logger.info(f"Loaded YOLO weights: {load_path} ({self.device})")

            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy, device=self.device, verbose=False, imgsz=640)
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False

    def detect(self, img, conf, iou, imgsz):
        if self.model is None:
            return []

        results = self.model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=self.device, verbose=False)
        model_names = getattr(self.model, 'names', {})
        ball_class_ids = set()
        
        iterator = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
        
        for c_id, name in iterator:
            if any(kw in str(name).lower() for kw in ['ball', 'cricket', 'sport']):
                ball_class_ids.add(c_id)
        
        detections = []
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                c = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                if ball_class_ids and c not in ball_class_ids:
                    continue
                confidence = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                detections.append((x, y, w, h, confidence, c))
        
        return detections

_global_yolo_detector = None

def get_global_yolo_detector(weights_path=None):
    global _global_yolo_detector
    default_weights = "weights/ball-yolov8s.pt"
    target_weights = weights_path if weights_path else default_weights
    if _global_yolo_detector is None:
        logger.info(f"Initializing YOLO detector with {target_weights}")
        _global_yolo_detector = YOLOBallDetector(target_weights)
    elif target_weights != _global_yolo_detector.model_path:
        logger.info(f"Reloading YOLO weights: {target_weights}")
        _global_yolo_detector.load_weights(target_weights)
    return _global_yolo_detector

def yolo_detect_ball(detector, frame):
    conf = DETECTION_CONFIG['conf_threshold']
    iou = DETECTION_CONFIG['iou_threshold']
    imgsz = GLOBAL_CONFIG['imgsz']

    return detector.detect(frame, conf=conf, iou=iou, imgsz=imgsz)
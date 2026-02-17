import os
import logging
import numpy as np
import torch
from ultralytics import YOLO
from global_config import GLOBAL_CONFIG
from BallDetection.config import DETECTION_CONFIG

logger = logging.getLogger(__name__)

class YOLOBallDetector:
    def __init__(self, model1_path=None, model2_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model1_path = model1_path if model1_path and os.path.isfile(model1_path) else DETECTION_CONFIG.get('model1_path', 'weights/ball-yolov8s.pt')
        self.model2_path = model2_path if model2_path and os.path.isfile(model2_path) else DETECTION_CONFIG.get('model2_path', 'weights/yolov8_cricket_ball2/weights/best.pt')
        try:
            self.model1 = YOLO(self.model1_path)
            logger.info(f"Loaded YOLO Model 1: {self.model1_path} ({self.device})")
        except Exception as e:
            logger.error(f"Failed to load Model 1: {e}")
            self.model1 = None
        try:
            self.model2 = YOLO(self.model2_path)
            logger.info(f"Loaded YOLO Model 2: {self.model2_path} ({self.device})")
        except Exception as e:
            logger.error(f"Failed to load Model 2: {e}")
            self.model2 = None

    def detect(self, img, conf, iou, imgsz):
        if self.model1 is None:
            return []
        results = self.model1.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=self.device, verbose=False)
        names = getattr(self.model1, 'names', {})
        if isinstance(names, dict):
            ball_ids = {c_id for c_id, name in names.items() if any(kw in str(name).lower() for kw in ['ball', 'cricket', 'sport'])}
        else:
            ball_ids = {c_id for c_id, name in enumerate(names) if any(kw in str(name).lower() for kw in ['ball', 'cricket', 'sport'])}
        detections = []
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                c = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                if ball_ids and c not in ball_ids:
                    continue
                conf_val = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                detections.append((x, y, w, h, conf_val, c))
        return detections

    def detect_roi(self, img_crop, offset_coords, conf, iou, imgsz):
        if self.model2 is None:
            return []
        results = self.model2.predict(img_crop, conf=conf, iou=iou, imgsz=imgsz, device=self.device, verbose=False)
        names = getattr(self.model2, 'names', {})
        if isinstance(names, dict):
            ball_ids = {c_id for c_id, name in names.items() if any(kw in str(name).lower() for kw in ['ball', 'cricket', 'sport'])}
        else:
            ball_ids = {c_id for c_id, name in enumerate(names) if any(kw in str(name).lower() for kw in ['ball', 'cricket', 'sport'])}
        detections = []
        x_offset, y_offset = offset_coords
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                c = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                if ball_ids and c not in ball_ids:
                    continue
                conf_val = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                # Map to global coordinates
                x, y, w, h = int(x1 + x_offset), int(y1 + y_offset), int(x2 - x1), int(y2 - y1)
                detections.append((x, y, w, h, conf_val, c))
        return detections

_global_yolo_detector = None

def get_global_yolo_detector():
    global _global_yolo_detector
    model1_path = DETECTION_CONFIG.get('model1_path')
    model2_path = DETECTION_CONFIG.get('model2_path')
    if (
        _global_yolo_detector is None
        or model1_path != getattr(_global_yolo_detector, 'model1_path', None)
        or model2_path != getattr(_global_yolo_detector, 'model2_path', None)
    ):
        logger.info(f"Initializing YOLO detector with Model 1: {model1_path}, Model 2: {model2_path}")
        _global_yolo_detector = YOLOBallDetector(model1_path, model2_path)
    return _global_yolo_detector

def yolo_detect_ball(detector, frame):
    conf = DETECTION_CONFIG['conf_threshold']
    iou = DETECTION_CONFIG['iou_threshold']
    imgsz = GLOBAL_CONFIG['imgsz']
    return detector.detect(frame, conf=conf, iou=iou, imgsz=imgsz)

def yolo_detect_ball_roi(detector, frame_crop, offset_coords):
    conf = DETECTION_CONFIG['conf_threshold']
    iou = DETECTION_CONFIG['iou_threshold']
    imgsz = GLOBAL_CONFIG['imgsz']
    return detector.detect_roi(frame_crop, offset_coords, conf=conf, iou=iou, imgsz=imgsz)
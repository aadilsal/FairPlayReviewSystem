import os
import logging
import numpy as np
import torch
import cv2
import math
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOBallDetector:
    """Ultralytics YOLOv8 wrapper with robust weight loading and device control.

    The public API remains similar: instantiate with a weights path and call
    `detect(img, conf, iou, imgsz)` which returns a list of (x,y,w,h,conf,cls).
    """
    def __init__(self, model_path='weights.pt', device: str = None):
        self.model = None
        self.model_path = None
        # Default to CPU for this environment unless device explicitly provided
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        # load the weights (will set self.model)
        self.load_weights(model_path)

    def load_weights(self, model_path: str):
        import os
        if model_path and os.path.exists(model_path):
            load_path = model_path
        else:
            logger.warning(f"Requested weights not found: {model_path}. Falling back to 'yolov8n.pt'.")
            load_path = 'yolov8n.pt'

        try:
            model = YOLO(load_path)
            self.model = model
            self.model_path = load_path
            logger.info(f"Loaded YOLO weights from: {load_path}")

            # Log classes
            names = getattr(self.model, 'names', None)
            logger.info(f"Model classes: {names}")
            if names is not None:
                lower_names = [str(v).lower() for v in names.values()] if isinstance(names, dict) else [str(n).lower() for n in names]
                # Check for ball-related classes (ball, sports ball, cricket ball, etc.)
                ball_keywords = ['ball', 'cricket', 'sport']
                found_ball_classes = [name for name in lower_names if any(kw in name for kw in ball_keywords)]
                if found_ball_classes:
                    logger.info(f"Ball-related classes found: {found_ball_classes}")
                else:
                    logger.warning("No ball-related classes found. Model may not detect balls. Consider training a custom model.")

            # Warm-up inference to detect obvious issues early
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, device=self.device, verbose=False)
                logger.info("Warm-up inference succeeded")
            except Exception as e:
                logger.warning(f"Warm-up inference failed: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            self.model = None
            return False

    def detect(self, img, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        """Run inference and return list of (x,y,w,h,confidence,cls_id).

        Keeps the same tuple format as earlier code so callers don't need to change.
        """
        detections = []
        if self.model is None:
            return detections

        results = self.model(img, conf=conf, iou=iou, imgsz=imgsz, device=self.device, verbose=False)
        
        # Get model class names for filtering
        model_names = getattr(self.model, 'names', {})
        ball_class_ids = []
        
        # Find ball-related classes (ball, sports ball, cricket ball, etc.)
        for cls_id, name in model_names.items() if isinstance(model_names, dict) else enumerate(model_names):
            name_lower = str(name).lower()
            if any(keyword in name_lower for keyword in ['ball', 'cricket', 'sport']):
                ball_class_ids.append(cls_id)
        
        if ball_class_ids:
            logger.debug(f"Using ball classes: {[model_names.get(i, i) for i in ball_class_ids]}")
        
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    # support both scalar and iterable cls
                    cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                except Exception:
                    cls_id = None
                
                # Filter: only accept ball-related classes if we found any
                if ball_class_ids and cls_id not in ball_class_ids:
                    continue
                
                try:
                    confidence = float(box.conf)
                except Exception:
                    confidence = 0.0
                try:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                except Exception:
                    continue
                detections.append((x, y, w, h, confidence, cls_id))

        # HSV FALLBACK DISABLED
        # After processing all results: if YOLO found nothing, run HSV-based fallback
        # if len(detections) == 0:
        #     try:
        #         hsv_det = _hsv_ball_detector(img)
        #         if hsv_det is not None:
        #             logger.info("YOLO returned no detections — using HSV fallback")
        #             detections.append(hsv_det)
        #     except Exception as e:
        #         logger.warning(f"HSV fallback failed: {e}")

        return detections


# HSV FALLBACK DISABLED
# def _hsv_ball_detector(bgr_img, min_area=300, circularity_thresh=0.35):
#     """Fallback HSV + shape detector for red/white cricket ball.
#
#     Returns a tuple (x,y,w,h,confidence,cls_id) or None if nothing found.
#     cls_id will be set to -1 for HSV fallback.
#     """
#     if bgr_img is None:
#         return None
#
#     img = bgr_img.copy()
#     # smooth to help with blur but keep edges
#     blur = cv2.GaussianBlur(img, (7, 7), 0)
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#
#     # red ranges (wrap-around)
#     lower_red1 = np.array([0, 70, 50])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([160, 70, 50])
#     upper_red2 = np.array([179, 255, 255])
#     mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     mask_red = cv2.bitwise_or(mask_r1, mask_r2)
#
#     # white: low saturation, high value
#     lower_white = np.array([0, 0, 180])
#     upper_white = np.array([179, 60, 255])
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)
#
#     mask = cv2.bitwise_or(mask_red, mask_white)
#
#     # morphological ops with elliptical kernel (round objects)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.dilate(mask, kernel, iterations=1)
#
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     best = None
#     best_score = 0.0
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < min_area:
#             continue
#         peri = cv2.arcLength(cnt, True)
#         if peri <= 0:
#             continue
#         circularity = 4.0 * math.pi * (area / (peri * peri))
#
#         (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
#         if radius <= 3:
#             continue
#
#         # normalize area by circle area
#         area_norm = area / (math.pi * (radius ** 2) + 1e-6)
#         # score blends circularity and area coverage
#         score = 0.6 * circularity + 0.4 * area_norm
#
#         if circularity >= circularity_thresh and score > best_score:
#             best_score = score
#             x_c, y_c, radius = float(x_c), float(y_c), float(radius)
#             x = int(max(0, x_c - radius))
#             y = int(max(0, y_c - radius))
#             w = int(min(bgr_img.shape[1] - x, 2 * radius))
#             h = int(min(bgr_img.shape[0] - y, 2 * radius))
#             confidence = float(min(1.0, score))
#             best = (x, y, w, h, confidence, -1)
#
#     return best

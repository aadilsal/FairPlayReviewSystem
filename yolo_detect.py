import os
import logging
import numpy as np
import torch
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
                if 'ball' not in lower_names:
                    logger.warning("'ball' not found in model class names. Check class mapping.")

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

        return detections

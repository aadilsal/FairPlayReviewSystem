import os
import logging
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLOBallDetector:
    """
    Ultralytics YOLOv8 wrapper.
    
    API:
      - __init__(model_path, device)
      - detect(img, conf, iou, imgsz) -> List[(x, y, w, h, conf, cls_id)]
    """
    def __init__(self, model_path='weights/ball-yolov8s.pt', device: str = None):
        self.model = None
        self.model_path = None
        
        # Default to CPU unless explicitly set or CUDA is available/requested
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load weights immediately
        self.load_weights(model_path)

    def load_weights(self, model_path: str):
        """Loads YOLO model from path with fallbacks and validation."""
        if model_path and os.path.exists(model_path):
            load_path = model_path
        else:
            logger.warning(f"Requested weights not found: {model_path}. Falling back to 'yolov8n.pt'.")
            load_path = 'yolov8n.pt'

        try:
            # Initialize Ultralytics YOLO
            model = YOLO(load_path)
            self.model = model
            self.model_path = load_path
            logger.info(f"Loaded YOLO weights from: {load_path} (Device: {self.device})")

            # Validate Classes
            names = getattr(self.model, 'names', {})
            if names:
                # Normalize names map to handle {0: 'ball'} or ['ball']
                if isinstance(names, dict):
                    name_list = [str(v).lower() for v in names.values()]
                else:
                    name_list = [str(n).lower() for n in names]
                
                logger.info(f"Model classes: {name_list}")
                
                # Check for relevant classes
                ball_keywords = ['ball', 'cricket', 'sport']
                found = [n for n in name_list if any(k in n for k in ball_keywords)]
                if found:
                    logger.info(f"Ball-related classes confirmed: {found}")
                else:
                    logger.warning("No ball-related classes found in model metadata.")
            
            # Warm-up Inference
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, device=self.device, verbose=False, imgsz=640)
                logger.info("Warm-up inference succeeded")
            except Exception as e:
                logger.warning(f"Warm-up inference failed: {e}")

            return True
        
        except Exception as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            self.model = None
            return False

    def detect(self, img, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        """
        Run inference.
        Returns: list of (x, y, w, h, confidence, cls_id)
        """
        detections = []
        if self.model is None:
            return detections

        # Run prediction
        # verbose=False keeps stdout clean
        results = self.model.predict(
            img, 
            conf=conf, 
            iou=iou, 
            imgsz=imgsz, 
            device=self.device, 
            verbose=False
        )
        
        # Identify ball-related class IDs dynamically to filter irrelevant objects
        model_names = getattr(self.model, 'names', {})
        ball_class_ids = set()
        
        if isinstance(model_names, dict):
            iterator = model_names.items()
        else:
            iterator = enumerate(model_names)
            
        for c_id, name in iterator:
            name_lower = str(name).lower()
            if any(kw in name_lower for kw in ['ball', 'cricket', 'sport']):
                ball_class_ids.add(c_id)
        
        if not ball_class_ids:
            # If no specific ball class found, we assume ALL classes are valid 
            # (e.g., custom model trained only on balls)
            pass 
        
        # Process Results
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            
            for box in boxes:
                # 1. Class ID
                cls_tensor = box.cls
                # Handle scalar vs tensor
                c = int(cls_tensor[0]) if hasattr(cls_tensor, '__len__') else int(cls_tensor)
                
                # Filter classes if we identified specific ball classes
                if ball_class_ids and c not in ball_class_ids:
                    continue
                
                # 2. Confidence
                conf_tensor = box.conf
                confidence = float(conf_tensor[0]) if hasattr(conf_tensor, '__len__') else float(conf_tensor)
                
                # 3. Bounding Box (xyxy -> xywh)
                # Ensure we move to CPU before converting to numpy
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to integer x, y, w, h
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # Append in the 6-tuple format expected by ball_detector.py
                detections.append((x, y, w, h, confidence, c))

        return detections
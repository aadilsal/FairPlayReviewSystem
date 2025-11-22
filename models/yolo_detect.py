import os
import cv2
import torch
import numpy as np


class YOLOBallDetector:
    """
    Compatibility wrapper that now loads a Roboflow RF-DETR (medium) model
    exported as `weights/weights.pt` and exposes the same `detect(img)` API
    used by the rest of the pipeline.

    NOTES:
    - This file preserves the original YOLO implementation commented out
      below for reference.
    - The new implementation attempts to handle several common exported
      formats (scripted TorchScript model, model object, or detection output
      dictionaries) and gracefully falls back if loading fails.

    Model details (for documentation):
    - Source: Roboflow project `cricket-dataset-z2wkt-nt696/1`
    - Type: RF-DETR (Medium)
    - Metrics: mAP@50=88.9%, Precision=92.0%, Recall=84.0%
    - Dataset version: 2025-11-09 3:22pm
    - Last updated: 11/10/25 06:11 AM
    - License: Apache-2.0

    The wrapper keeps the public contract:
      - `load_weights(model_path)` -> bool
      - `detect(img)` -> list of (x, y, w, h, confidence)

    """

    # Detection thresholds (configured per request)
    CONFIDENCE_THRESHOLD = 0.56
    IOU_THRESHOLD = 0.50
    VIS_OPACITY = 0.75

    def __init__(self, model_path='yolov8n.pt'):
        self.model = None
        self.model_path = None
        self.is_torchscript = False
        # Try to load the provided path; if missing, we keep model=None
        self.load_weights(model_path)

    def load_weights(self, model_path):
        """Attempt to load a Roboflow RF-DETR model from `model_path`.

        Returns True on success, False otherwise. The loader tries several
        strategies to be robust to different export formats.
        """
        if not model_path:
            return False

        if not os.path.exists(model_path):
            print(f"[WARN] Requested weights not found: {model_path}")
            return False

        try:
            # 1) Try loading as TorchScript (most robust if Roboflow exported a
            # scripted model)
            try:
                self.model = torch.jit.load(model_path, map_location='cpu')
                self.model.eval()
                self.is_torchscript = True
                self.model_path = model_path
                print(f"[INFO] Loaded TorchScript RF-DETR model from: {model_path}")
                return True
            except Exception:
                # Not TorchScript - fall through
                pass

            # 2) Try loading state dict / checkpoint and keep raw object
            ckpt = torch.load(model_path, map_location='cpu')
            # If the checkpoint is a scripted module, use it directly
            if isinstance(ckpt, torch.jit.ScriptModule) or hasattr(ckpt, 'forward'):
                self.model = ckpt
                self.model.eval()
                self.model_path = model_path
                print(f"[INFO] Loaded RF-DETR model object from: {model_path}")
                return True

            # Otherwise, if it's a dict, stash it so advanced users can hook it
            if isinstance(ckpt, dict):
                # We don't have the model architecture here; keep checkpoint for
                # potential future loading. Inform the user and return False.
                print(f"[WARN] Checkpoint appears to be a state_dict. Architecture required to load: {model_path}")
                self.model = None
                return False

            # Unknown format
            print(f"[ERROR] Unrecognized RF-DETR model format: {model_path}")
            self.model = None
            return False

        except Exception as e:
            print(f"[ERROR] Could not load RF-DETR model: {e}")
            self.model = None
            return False

    def _preprocess(self, img):
        # Convert BGR (cv2) image to RGB tensor, resize preserving aspect ratio
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        # RF-DETR typically accepts images scaled to a size; we'll center-pad to
        # the largest side and return a 3xHxW tensor.
        h, w, _ = img_float.shape
        max_side = max(h, w)
        pad_img = np.zeros((max_side, max_side, 3), dtype=np.float32)
        pad_img[:h, :w] = img_float
        tensor = torch.from_numpy(pad_img).permute(2, 0, 1).unsqueeze(0)
        return tensor, (w, h)

    def _postprocess_outputs(self, outputs, orig_size):
        """Attempt to normalize different possible output formats into a
        list of (x,y,w,h,conf) with coordinates in original image space.
        """
        detections = []
        ow, oh = orig_size

        # Common: model returns a dict with 'boxes', 'scores', 'labels'
        if isinstance(outputs, dict):
            boxes = outputs.get('boxes')
            scores = outputs.get('scores')
            labels = outputs.get('labels', None)
            if boxes is not None and scores is not None:
                boxes = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.array(boxes)
                scores = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
                for i, box in enumerate(boxes):
                    score = float(scores[i])
                    if score < self.CONFIDENCE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = box
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    detections.append((x, y, w, h, score))
                return detections

        # Common PyTorch/DETR: list of dicts
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if not isinstance(item, dict):
                    continue
                boxes = item.get('boxes')
                scores = item.get('scores')
                if boxes is None or scores is None:
                    continue
                boxes = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.array(boxes)
                scores = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
                for i, box in enumerate(boxes):
                    score = float(scores[i])
                    if score < self.CONFIDENCE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = box
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    detections.append((x, y, w, h, score))
            return detections

        # If outputs is a Tensor Nx6 or Nx5 ([x1,y1,x2,y2,score] or with label)
        if hasattr(outputs, 'cpu') and isinstance(outputs, torch.Tensor):
            arr = outputs.cpu().numpy()
            if arr.ndim == 2 and arr.shape[1] >= 5:
                for row in arr:
                    x1, y1, x2, y2, score = row[:5]
                    if score < self.CONFIDENCE_THRESHOLD:
                        continue
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    detections.append((x, y, w, h, float(score)))
                return detections

        # Unknown output format
        return detections

    def detect(self, img):
        """Run detection on an image and return list of (x,y,w,h,conf).

        This method attempts to call the loaded RF-DETR model and normalize
        outputs to the same simple format used elsewhere in the codebase.
        If the model is not loaded or outputs are unrecognized, returns [].
        """
        if self.model is None:
            return []

        try:
            tensor, orig_size = self._preprocess(img)
            with torch.no_grad():
                if self.is_torchscript:
                    outputs = self.model(tensor)
                else:
                    # Some exported models expect the raw tensor; others a dict
                    try:
                        outputs = self.model(tensor)
                    except TypeError:
                        outputs = self.model.forward(tensor)

            detections = self._postprocess_outputs(outputs, orig_size)
            return detections
        except Exception as e:
            print(f"[ERROR] RF-DETR detection failed: {e}")
            return []


# ============================================================================
# OLD YOLO IMPLEMENTATION - Replaced with Roboflow RF-DETR model on 2025-11-16
# The original YOLO-based class is preserved below for reference and can be
# re-enabled if required. Keeping the old implementation commented-out makes
# it easier to switch back or port parts of it.
#
"""
import os
import cv2
from ultralytics import YOLO


class _YOLO_Old_Impl:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = None
        self.model_path = None
        self.load_weights(model_path)

    def load_weights(self, model_path):
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"[INFO] Loaded YOLO weights from: {model_path}")
            return True
        fallback = 'yolov8n.pt'
        try:
            self.model = YOLO(fallback)
            self.model_path = fallback
            print(f"[WARN] Requested weights not found. Loaded fallback: {fallback}")
            return True
        except Exception as e:
            print(f"[ERROR] Could not load YOLO model: {e}")
            self.model = None
            return False

    def detect(self, img):
        detections = []
        if self.model is None:
            return detections
        results = self.model(img)
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is not None:
                for box in boxes:
                    try:
                        if int(box.cls) == 0:
                            confidence = float(box.conf)
                            coords = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = coords
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                            detections.append((x, y, w, h, confidence))
                    except Exception:
                        continue
        return detections

"""

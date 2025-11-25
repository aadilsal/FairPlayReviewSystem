import cv2
import logging
import os
from yolo_detect import YOLOBallDetector
from preprocessing import preprocess_frame

logger = logging.getLogger(__name__)

# Initialize detectors once but allow runtime weight replacement
_yolo_detector = None

DETECTION_CONFIG = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'imgsz': 640,
    'min_area': 100,
    'max_area': 3000,
    'aspect_ratio_min': 0.7,
    'aspect_ratio_max': 1.3,
    'ball_color': 'white',
    'enable_preprocessing': True,
}


def update_detection_config(**kwargs):
    DETECTION_CONFIG.update(kwargs)
    logger.info(f"Updated detection config: {kwargs}")


def get_yolo_detector(weights_path=None, device: str = None):
    global _yolo_detector
    # prefer provided path, else default to runtime config
    from weights_config import YOLO_BALL_WEIGHTS
    weights = weights_path if weights_path is not None else YOLO_BALL_WEIGHTS

    if _yolo_detector is None:
        logger.info(f"Initializing YOLO detector with weights: {weights} (device={device})")
        _yolo_detector = YOLOBallDetector(weights, device=device)
    elif weights is not None and os.path.exists(weights) and weights != _yolo_detector.model_path:
        logger.info(f"Reloading YOLO weights: {weights}")
        _yolo_detector.load_weights(weights)
    return _yolo_detector

def detect_ball_on_frame(frame, yolo_weights=None, debug=False,
                         enable_preprocessing=None, ball_color=None,
                         conf_threshold=None, iou_threshold=None, imgsz=None):
    """
    Draw ball detection (YOLO or color-based) on top of the input frame.
    Backwards-compatible: still returns (frame_with_ball, detected:bool).
    New optional args allow preprocessing and tuning.
    """
    frame_with_ball = frame.copy()
    found = False

    # Resolve config defaults
    if enable_preprocessing is None:
        enable_preprocessing = DETECTION_CONFIG.get('enable_preprocessing', True)
    if ball_color is None:
        ball_color = DETECTION_CONFIG.get('ball_color', 'white')
    if conf_threshold is None:
        conf_threshold = DETECTION_CONFIG.get('conf_threshold', 0.25)
    if iou_threshold is None:
        iou_threshold = DETECTION_CONFIG.get('iou_threshold', 0.45)
    if imgsz is None:
        imgsz = DETECTION_CONFIG.get('imgsz', 640)

    # PREPROCESSING
    processed_frame = frame
    debug_info = {}
    if enable_preprocessing:
        processed_frame, debug_info = preprocess_frame(frame, ball_color=ball_color)
        logger.debug(f"Preprocessing applied: {debug_info}")

    # Detection
    detector = get_yolo_detector(yolo_weights)
    yolo_detections = detector.detect(processed_frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz)

    filtered = []
    # Draw raw YOLO detections faintly for debugging
    for det in yolo_detections:
        if len(det) == 6:
            x, y, w, h, confidence, cls_id = det
        else:
            x, y, w, h, confidence = det
            cls_id = None
        if debug:
            cv2.rectangle(frame_with_ball, (x, y), (x + w, y + h), (200, 200, 0), 1)
            label = f"cls:{cls_id} {confidence:.2f}" if cls_id is not None else f"{confidence:.2f}"
            cv2.putText(frame_with_ball, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

    for det in yolo_detections:
        if len(det) == 6:
            (x, y, w, h, confidence, cls_id) = det
        else:
            (x, y, w, h, confidence) = det
            cls_id = None
        aspect = (w / h) if h > 0 else 0
        area = w * h
        if DETECTION_CONFIG['aspect_ratio_min'] < aspect < DETECTION_CONFIG['aspect_ratio_max'] and DETECTION_CONFIG['min_area'] < area < DETECTION_CONFIG['max_area']:
            filtered.append((x, y, w, h, confidence))

    if filtered:
        for (x, y, w, h, confidence) in filtered:
            cv2.rectangle(frame_with_ball, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_with_ball, f"Ball {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        found = True
    else:
        # Prefer the HSV/shape fallback implemented inside YOLOBallDetector (cls_id == -1)
        hsv_used = False
        for det in yolo_detections:
            if len(det) == 6:
                x, y, w, h, confidence, cls_id = det
            else:
                x, y, w, h, confidence = det
                cls_id = None
            if cls_id == -1:
                # Draw HSV fallback detection in green
                cv2.rectangle(frame_with_ball, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_with_ball, f"HSV Ball {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                logger.debug("Using HSV fallback detection from YOLO detector")
                found = True
                hsv_used = True
                break

        # If no HSV fallback present in YOLO detections, keep existing behaviour (no detection)

    return frame_with_ball, found
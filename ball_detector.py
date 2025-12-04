import cv2
import logging
import os
import numpy as np
from yolo_detect import YOLOBallDetector
from preprocessing import preprocess_frame

logger = logging.getLogger(__name__)

# Initialize detectors once but allow runtime weight replacement
_yolo_detector = None

DETECTION_CONFIG = {
    'conf_threshold': 0.4,
    'iou_threshold': 0.45,
    'imgsz': 640,
    'min_area': 100,
    'max_area': 3000,
    'aspect_ratio_min': 0.7,
    'aspect_ratio_max': 1.3,
    'ball_color': 'white',
    'enable_preprocessing': True,
    'enable_color_filter': True,
    'color_threshold': 0.3,  # 30% of pixels must match ball color
    'enable_motion_tracking': True,
    'min_velocity': 3,  # pixels per frame
    'max_trajectory_deviation': 50,  # pixels
}

# Motion tracking state
class BallMotionTracker:
    def __init__(self, max_history=5):
        self.positions = []  # [(center_x, center_y, frame_idx), ...]
        self.max_history = max_history
    
    def reset(self):
        self.positions = []
    
    def validate_detection(self, center, frame_idx):
        """Check if detection follows consistent motion pattern."""
        if len(self.positions) < 2:
            self.positions.append((center[0], center[1], frame_idx))
            return True
        
        # Calculate velocity
        prev_x, prev_y, prev_frame = self.positions[-1]
        dt = frame_idx - prev_frame
        if dt == 0:
            return True  # Same frame
        
        velocity_x = (center[0] - prev_x) / dt
        velocity_y = (center[1] - prev_y) / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Balls should move with minimum velocity
        min_vel = DETECTION_CONFIG.get('min_velocity', 3)
        if speed < min_vel:
            logger.debug(f"Rejected: velocity too low ({speed:.1f} < {min_vel})")
            return False
        
        # Check trajectory smoothness if we have enough history
        if len(self.positions) >= 3:
            # Calculate deviation from expected path
            last_3 = self.positions[-3:]
            # Simple linear prediction
            expected_x = 2 * last_3[-1][0] - last_3[-2][0]
            expected_y = 2 * last_3[-1][1] - last_3[-2][1]
            deviation = np.sqrt((center[0] - expected_x)**2 + (center[1] - expected_y)**2)
            
            max_dev = DETECTION_CONFIG.get('max_trajectory_deviation', 50)
            if deviation > max_dev:
                logger.debug(f"Rejected: trajectory deviation too high ({deviation:.1f} > {max_dev})")
                return False
        
        # Update history
        self.positions.append((center[0], center[1], frame_idx))
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
        
        return True

_motion_tracker = BallMotionTracker()

def is_ball_colored(frame, x, y, w, h, ball_color='white'):
    """Validate if detection region has ball-like colors."""
    try:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if ball_color == 'white':
            # White: high value (brightness), low saturation
            mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        elif ball_color == 'red':
            # Red: two ranges in HSV (wraps around at 180)
            mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Default: accept any color
            return True
        
        # Calculate percentage of pixels matching ball color
        matching_pixels = cv2.countNonZero(mask)
        total_pixels = w * h
        color_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0
        
        threshold = DETECTION_CONFIG.get('color_threshold', 0.3)
        return color_ratio >= threshold
    except Exception as e:
        logger.warning(f"Color validation failed: {e}")
        return True  # If validation fails, don't reject


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
                         conf_threshold=None, iou_threshold=None, imgsz=None,
                         frame_idx=0):
    """
    Draw ball detection (YOLO or color-based) on top of the input frame.
    Backwards-compatible: still returns (frame_with_ball, detected:bool).
    New optional args allow preprocessing and tuning.
    Includes color validation and motion tracking filters.
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
        
        # Filter 1: Aspect ratio and area
        aspect = (w / h) if h > 0 else 0
        area = w * h
        if not (DETECTION_CONFIG['aspect_ratio_min'] < aspect < DETECTION_CONFIG['aspect_ratio_max']):
            if debug:
                logger.debug(f"Rejected: aspect ratio {aspect:.2f} outside range")
            continue
        if not (DETECTION_CONFIG['min_area'] < area < DETECTION_CONFIG['max_area']):
            if debug:
                logger.debug(f"Rejected: area {area} outside range")
            continue
        
        # Filter 2: Color validation
        if DETECTION_CONFIG.get('enable_color_filter', True):
            if not is_ball_colored(frame, x, y, w, h, ball_color):
                if debug:
                    logger.debug(f"Rejected: color validation failed at ({x},{y})")
                continue
        
        # Filter 3: Motion tracking
        if DETECTION_CONFIG.get('enable_motion_tracking', True):
            center = (x + w // 2, y + h // 2)
            if not _motion_tracker.validate_detection(center, frame_idx):
                if debug:
                    logger.debug(f"Rejected: motion validation failed at ({x},{y})")
                continue
        
        # All filters passed!
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
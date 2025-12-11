import cv2
import logging
import os
import numpy as np
from typing import Tuple, Optional, List

# 1. IMPORT PREPROCESSING (The missing link)
try:
    from preprocessing import preprocess_frame
except ImportError:
    print("[WARN] preprocessing.py not found. Using raw frames.")
    preprocess_frame = None

# Ensure you have this file or adjust import
from yolo_detect import YOLOBallDetector 

logger = logging.getLogger(__name__)

# Initialize detectors once
_yolo_detector = None

# 2. RELAXED CONFIGURATION (Optimized for Cricket/Sports)
DETECTION_CONFIG = {
    'conf_threshold': 0.2,
    'iou_threshold': 0.45,
    'imgsz': 640,
    'min_area': 20,               # ✅ FIX: Reduced to 20 to catch small/far balls
    'max_area': 4000,
    'aspect_ratio_min': 0.5,      # ✅ FIX: Lowered to 0.5 for motion blur
    'aspect_ratio_max': 2.5,      # ✅ FIX: Increased to 2.5 for elongated blur
    'ball_color': 'red',          # ✅ FIX: Default to 'red' (change to 'white' if needed)
    'enable_preprocessing': True, # ✅ FIX: Enabled by default
    'enable_color_filter': False, # ✅ FIX: Disabled initially to avoid false negatives
    'color_threshold': 0.2,
    'enable_motion_tracking': True,
    'min_velocity': 1,            # ✅ FIX: Low velocity allowed
    'max_trajectory_deviation': 100,
    'use_optical_flow': True,     # ✅ NEW: Enable optical flow tracking by default
}

# ---------------------------------------------------------
# HELPERS & TRACKER
# ---------------------------------------------------------

class BallMotionTracker:
    def __init__(self, max_history=5):
        self.positions = []  # [(center_x, center_y, frame_idx), ...]
        self.max_history = max_history
    
    def reset(self):
        self.positions = []
    
    def validate_detection(self, center, frame_idx):
        """Check if detection follows consistent motion pattern."""
        if len(self.positions) < 1:
            self.positions.append((center[0], center[1], frame_idx))
            return True
        
        # Calculate velocity
        prev_x, prev_y, prev_frame = self.positions[-1]
        dt = frame_idx - prev_frame
        
        # If pipeline doesn't pass frame_idx, dt is 0. 
        if dt == 0:
            return True 
        
        velocity_x = (center[0] - prev_x) / dt
        velocity_y = (center[1] - prev_y) / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Relaxed velocity check
        min_vel = DETECTION_CONFIG.get('min_velocity', 1)
        # Note: We barely reject based on low speed anymore to be safe
        
        # Update history
        self.positions.append((center[0], center[1], frame_idx))
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
        
        return True

_motion_tracker = BallMotionTracker()

class OpticalFlowBallTracker:
    """
    Tracks ball using Lucas-Kanade optical flow when YOLO fails.
    Stores last detected position and previous frame for flow calculation.
    Predicts position for up to 5 consecutive frames without YOLO confirmation.
    """
    def __init__(self, max_consecutive_predictions=5):
        self.last_center = None  # (x, y) of ball center
        self.last_w = None       # Width of last detected box
        self.last_h = None       # Height of last detected box
        self.prev_frame = None   # Previous grayscale frame
        self.consecutive_predictions = 0
        self.max_consecutive = max_consecutive_predictions
        # Optical flow parameters (Lucas-Kanade)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def update_detection(self, frame, ball_center, w, h):
        """
        Update tracker with successful YOLO detection.
        Saves position, box size, and current frame (converted to grayscale).
        Resets prediction counter.
        """
        self.last_center = ball_center
        self.last_w = w
        self.last_h = h
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.consecutive_predictions = 0
        logger.debug(f"[OpticalFlow] Updated with detection at {ball_center}, size {w}x{h}")
    
    def predict_position(self, current_frame):
        """
        Predict ball position using optical flow from prev_frame to current_frame.
        Returns (x, y, confidence) if successful, else None.
        Stops if: no previous data, poor flow quality, out of bounds, or max consecutive predictions reached.
        """
        if self.last_center is None or self.prev_frame is None:
            return None
        
        if self.consecutive_predictions >= self.max_consecutive:
            logger.debug("[OpticalFlow] Max consecutive predictions reached, stopping")
            self.reset()
            return None
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_points = np.array([self.last_center], dtype=np.float32).reshape(-1, 1, 2)
        
        # Compute optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, current_gray, prev_points, None, **self.lk_params)
        
        # Check quality: status must be 1 (good tracking), and error low (threshold: 10 for reasonable quality)
        if status[0] != 1 or error[0] > 10:
            logger.debug(f"[OpticalFlow] Poor tracking quality: status={status[0]}, error={error[0]}")
            self.reset()
            return None
        
        predicted_center = (int(next_points[0][0][0]), int(next_points[0][0][1]))
        
        # Check if within frame bounds (with margin)
        h, w = current_frame.shape[:2]
        margin = 10
        if not (margin <= predicted_center[0] <= w - margin and margin <= predicted_center[1] <= h - margin):
            logger.debug(f"[OpticalFlow] Predicted position out of bounds: {predicted_center}")
            self.reset()
            return None
        
        # Update for next prediction
        self.last_center = predicted_center
        self.prev_frame = current_gray
        self.consecutive_predictions += 1
        
        # Confidence: 1.0 for good prediction (could be scaled by error if needed)
        confidence = 1.0
        logger.debug(f"[OpticalFlow] Predicted position: {predicted_center}, consecutive={self.consecutive_predictions}")
        return predicted_center[0], predicted_center[1], confidence
    
    def reset(self):
        """Clear all tracking state."""
        self.last_center = None
        self.last_w = None
        self.last_h = None
        self.prev_frame = None
        self.consecutive_predictions = 0
        logger.debug("[OpticalFlow] Tracker reset")

# Global optical flow tracker
_optical_flow_tracker = OpticalFlowBallTracker()

def is_ball_colored(frame, x, y, w, h, ball_color='white'):
    """Validate if detection region has ball-like colors."""
    try:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if ball_color == 'white':
            mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255)) # Relaxed white
        elif ball_color == 'red':
            mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            return True
        
        matching_pixels = cv2.countNonZero(mask)
        total_pixels = w * h
        color_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0
        
        threshold = DETECTION_CONFIG.get('color_threshold', 0.2)
        return color_ratio >= threshold
    except Exception as e:
        print(f"[WARN] Color validation failed: {e}")
        return True


def is_shoe_like(img: np.ndarray, bbox: Tuple[int, int, int, int], elongation_thresh: float = 3.0, bottom_of_frame_margin: int = 20) -> bool:
    """Heuristic to reject detections that look like shoes."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    # Only check elongation if it's at the very bottom of the frame
    img_h = img.shape[0]
    if (img_h - y2) <= bottom_of_frame_margin:
        elongation = w / h
        if elongation > elongation_thresh:
            return True

    return False


def is_ball_circular(img: np.ndarray, bbox: Tuple[int, int, int, int], circularity_thresh: float = 0.4) -> bool:
    """Estimate circularity. Lower threshold allowed for motion blur."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return False
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return False
        
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5: return False 
    
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0: return False
        
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity >= circularity_thresh


def get_yolo_detector(weights_path=None):
    global _yolo_detector
    default_weights = "weights/ball-yolov8s.pt" 

    if _yolo_detector is None:
        target_weights = weights_path if weights_path else default_weights
        print(f"[INFO] Initializing YOLO detector with {target_weights}")
        _yolo_detector = YOLOBallDetector(target_weights)
    elif weights_path is not None and weights_path != _yolo_detector.model_path:
        print(f"[INFO] Reloading YOLO weights: {weights_path}")
        _yolo_detector.load_weights(weights_path)
        
    return _yolo_detector

# ---------------------------------------------------------
# MAIN DETECTOR FUNCTION
# ---------------------------------------------------------
def detect_ball_on_frame(
    frame, 
    yolo_weights=None, 
    debug=False,         
    enable_preprocessing=None, 
    ball_color=None,
    conf_threshold=None, 
    iou_threshold=None, 
    imgsz=None,
    frame_idx=0,
    enable_optical_flow=None  # NEW: Parameter to enable/disable optical flow
):
    """
    Returns (frame, ball_info)
    ball_info is { "box": [x,y,w,h], "conf": float, "source": "yolo"|"optical_flow" } or None
    """
    # 1. Config Resolution
    if ball_color is None: ball_color = DETECTION_CONFIG.get('ball_color', 'red')
    if conf_threshold is None: conf_threshold = DETECTION_CONFIG.get('conf_threshold', 0.2)
    if iou_threshold is None: iou_threshold = DETECTION_CONFIG.get('iou_threshold', 0.45)
    if imgsz is None: imgsz = DETECTION_CONFIG.get('imgsz', 640)
    if enable_preprocessing is None: enable_preprocessing = DETECTION_CONFIG.get('enable_preprocessing', True)
    if enable_optical_flow is None: enable_optical_flow = DETECTION_CONFIG.get('use_optical_flow', True)

    # 2. PREPROCESSING (The Key Fix)
    detection_frame = frame
    if enable_preprocessing and preprocess_frame is not None:
        try:
            # Enhance contrast/sharpness so YOLO sees the ball better
            detection_frame, _ = preprocess_frame(frame, ball_color=ball_color)
        except Exception as e:
            if debug: print(f"[WARN] Preprocessing failed: {e}")

    # 3. Run Inference on ENHANCED frame
    detector = get_yolo_detector(yolo_weights)
    yolo_detections = detector.detect(detection_frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz)

    filtered = []

    # 4. Filtering Loop
    for det in yolo_detections:
        if len(det) == 6:
            (x, y, w, h, confidence, cls_id) = det
        else:
            (x, y, w, h, confidence) = det
        
        # --- Filter A: Aspect Ratio & Area ---
        aspect = (w / h) if h > 0 else 0
        area = w * h
        
        if area < DETECTION_CONFIG['min_area'] or area > DETECTION_CONFIG['max_area']:
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Area {area}")
            continue

        if not (DETECTION_CONFIG['aspect_ratio_min'] < aspect < DETECTION_CONFIG['aspect_ratio_max']):
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Aspect Ratio {aspect:.2f}")
            continue
        
        x1, y1, x2, y2 = x, y, x + w, y + h

        # --- Filter B: Shoe-like rejection ---
        if is_shoe_like(frame, (x1, y1, x2, y2)):
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Shoe-like")
            continue

        # --- Filter C: Circularity check ---
        # Note: We relaxed the threshold to 0.4
        if not is_ball_circular(frame, (x1, y1, x2, y2)):
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Non-circular")
            continue

        # --- Filter D: Color validation ---
        if DETECTION_CONFIG.get('enable_color_filter', False):
            if not is_ball_colored(frame, x, y, w, h, ball_color):
                if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Color mismatch ({ball_color})")
                continue

        # --- Filter E: Motion tracking ---
        if DETECTION_CONFIG.get('enable_motion_tracking', True):
            center = (x + w // 2, y + h // 2)
            if not _motion_tracker.validate_detection(center, frame_idx):
                if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Motion trajectory")
                continue
        
        # Accepted
        filtered.append((x, y, w, h, confidence))

    # 5. Prepare Result from YOLO
    ball_info = None
    if filtered:
        # Sort by confidence (descending)
        filtered.sort(key=lambda x: x[4], reverse=True)
        best_det = filtered[0]
        
        ball_info = {
            "box": [best_det[0], best_det[1], best_det[2], best_det[3]],
            "conf": float(best_det[4]),
            "source": "yolo"
        }
        
        # Update optical flow tracker with successful detection
        center = (best_det[0] + best_det[2] // 2, best_det[1] + best_det[3] // 2)
        _optical_flow_tracker.update_detection(frame, center, best_det[2], best_det[3])
    
    # 6. Fallback to Optical Flow if YOLO failed and enabled
    elif enable_optical_flow:
        predicted = _optical_flow_tracker.predict_position(frame)
        if predicted:
            px, py, p_conf = predicted
            # Use last known box size, center at predicted position
            if _optical_flow_tracker.last_w is not None and _optical_flow_tracker.last_h is not None:
                x = px - _optical_flow_tracker.last_w // 2
                y = py - _optical_flow_tracker.last_h // 2
                ball_info = {
                    "box": [x, y, _optical_flow_tracker.last_w, _optical_flow_tracker.last_h],
                    "conf": -1.0,  # Negative to indicate prediction
                    "source": "optical_flow"
                }
                if debug: print(f"[DEBUG] Frame {frame_idx}: Using optical flow prediction at ({px}, {py})")

    return frame, ball_info
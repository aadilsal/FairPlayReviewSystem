import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from BallDetection.utils.config import FILTERS_CONFIG
from BallDetection.pipeline.trajectory import predict_position, TrajectoryModel

logger = logging.getLogger(__name__)

def is_shoe_like(img, bbox):
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    img_h = img.shape[0]
    return (img_h - y2) <= 20 and (w / h) > 3.0


def is_ball_circular(img, bbox):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)  
    if area < 5:
        return False
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity >= 0.4


def is_ball_colored(frame, x, y, w, h):
    ball_color = FILTERS_CONFIG['ball_color']
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if ball_color == 'white':
        mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    elif ball_color == 'red':
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        return True
    matching_pixels = cv2.countNonZero(mask)
    total_pixels = w * h
    color_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0
    return color_ratio >= FILTERS_CONFIG.get('color_threshold', 0.2)


def corridor_check(detection_xy: Tuple[float, float], trajectory_model: TrajectoryModel, 
                   frame_idx: int, corridor_width: float) -> bool:
    """
    Hard gate: Returns True if the detection falls within ±corridor_width pixels 
    of the physically predicted trajectory position.
    """
    pred_pos = predict_position(trajectory_model, frame_idx)
    if pred_pos is None:
        return False
        
    dist = np.sqrt((detection_xy[0] - pred_pos[0])**2 + (detection_xy[1] - pred_pos[1])**2)
    return dist <= corridor_width


def motion_blur_alignment_check():
    """Placeholder for Phase 3: Alignment check between bbox aspect ratio and velocity vector."""
    # TODO: Implement in Stage 12 optimization
    pass


def filter_ball_detection(img, bbox):
    """Applies all configured filters to a single detection."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    area = w * h
    aspect = (w / h) if h > 0 else 0

    # 1. Area Filter
    if FILTERS_CONFIG.get('enable_area_filter'):
        if area < FILTERS_CONFIG['min_area'] or area > FILTERS_CONFIG['max_area']:
            return False
            
    # 2. Aspect Ratio Filter
    if FILTERS_CONFIG.get('enable_aspect_ratio_filter'):
        if not (FILTERS_CONFIG['aspect_ratio_min'] < aspect < FILTERS_CONFIG['aspect_ratio_max']):
            return False

    # 3. Shoe Filter (Heuristic)
    if FILTERS_CONFIG.get('enable_shoe_filter'):
        if is_shoe_like(img, (x1, y1, x2, y2)):
            return False
            
    # 4. Circularity Filter (Computer Vision)
    if FILTERS_CONFIG.get('enable_circularity_filter'):
        if not is_ball_circular(img, (x1, y1, x2, y2)):
            return False
            
    # 5. Color Filter (HSV Analysis)
    if FILTERS_CONFIG.get('enable_color_filter'):
        if not is_ball_colored(img, x1, y1, w, h):
            return False
            
    return True


def filter_and_select_ball_detection(frame, detections):
    """Filters multiple detections and selects the best one based on confidence."""
    filtered = []
    
    if len(detections) > 0:
        logger.debug(f"[FILTER] Received {len(detections)} raw detections from YOLO.")

    for det in detections:
        x, y, w, h, confidence = det[:5]
        bbox = (x, y, x + w, y + h)
        
        if filter_ball_detection(frame, bbox):
            filtered.append({
                'box': [float(x), float(y), float(w), float(h)], 
                'conf': float(confidence), 
                'source': 'yolo'
            })
            
    if filtered:
        filtered.sort(key=lambda x: x['conf'], reverse=True)
        return filtered[0]
    
    if len(detections) > 0 and not filtered:
        logger.warning(f"[FILTER] All {len(detections)} YOLO detections were rejected by filters.")
        
    return None

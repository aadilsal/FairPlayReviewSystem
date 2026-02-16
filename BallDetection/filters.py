import numpy as np
import cv2
from BallDetection.config import DETECTION_CONFIG

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


def is_ball_colored(frame, x, y, w, h, ball_color='white'):
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
    return color_ratio >= DETECTION_CONFIG.get('color_threshold', 0.2)


def filter_ball_detection(img, bbox, ball_color):
    if is_shoe_like(img, bbox):
        return False
    if not is_ball_circular(img, bbox):
        return False
    if DETECTION_CONFIG.get('enable_color_filter', False):
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if not is_ball_colored(img, x1, y1, w, h, ball_color):
            return False
    return True


def filter_and_select_ball_detection(frame, detections, ball_color):
    filtered = []
    for det in detections:
        x, y, w, h, confidence = det[:5]
        aspect = (w / h) if h > 0 else 0
        area = w * h
        if area < DETECTION_CONFIG['min_area'] or area > DETECTION_CONFIG['max_area']:
            continue
        if not (DETECTION_CONFIG['aspect_ratio_min'] < aspect < DETECTION_CONFIG['aspect_ratio_max']):
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox = (x1, y1, x2, y2)
        if not filter_ball_detection(frame, bbox, ball_color):
            continue
        filtered.append({'box': [x, y, w, h], 'conf': float(confidence), 'source': 'yolo'})
    if filtered:
        filtered.sort(key=lambda x: x['conf'], reverse=True)
        return filtered[0]
    return None

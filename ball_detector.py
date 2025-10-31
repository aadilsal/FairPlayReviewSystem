import cv2
from yolo_detect import YOLOBallDetector
from ball_tracker import ball_detect
from cvzone.ColorModule import ColorFinder
import os

# Initialize detectors once but allow runtime weight replacement
_yolo_detector = None
def get_yolo_detector(weights_path=None):
    global _yolo_detector
    if _yolo_detector is None:
        # prefer provided path, else default (wrapped in YOLOBallDetector)
        weights = weights_path if weights_path is not None else 'outputs/yolov8_cricket_ball2/weights/best.pt'
        _yolo_detector = YOLOBallDetector(weights)
    elif weights_path is not None and os.path.exists(weights_path) and weights_path != _yolo_detector.model_path:
        _yolo_detector.load_weights(weights_path)
    return _yolo_detector


color_finder = ColorFinder(False)
hsv_vals = {
    "hmin": 10, "smin": 44, "vmin": 192,
    "hmax": 125, "smax": 114, "vmax": 255,
}

def detect_ball_on_frame(frame, yolo_weights=None):
    """
    Draw ball detection (YOLO or color-based) on top of the input frame.
    Returns (frame_with_ball, detected:bool)
    """
    frame_with_ball = frame.copy()
    found = False

    # Try YOLO detection with size/aspect filtering
    detector = get_yolo_detector(yolo_weights)
    yolo_detections = detector.detect(frame)
    filtered = []
    for (x, y, w, h, confidence) in yolo_detections:
        aspect = w / h if h > 0 else 0
        area = w * h
        if 0.7 < aspect < 1.3 and 100 < area < 3000:
            filtered.append((x, y, w, h, confidence))
    if filtered:
        for (x, y, w, h, confidence) in filtered:
            cv2.rectangle(frame_with_ball, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_with_ball, f"Ball {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        found = True
    else:
        # Fallback to color-based detection
        img_contours, x, y = ball_detect(frame, color_finder, hsv_vals)
        if img_contours is not None:
            overlay = frame_with_ball.copy()
            mask = cv2.cvtColor(img_contours, cv2.COLOR_BGR2GRAY)
            overlay[mask > 0] = img_contours[mask > 0]
            frame_with_ball = overlay
            found = True

    return frame_with_ball, found
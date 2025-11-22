
import cv2
import os
from config.paths import YOLO_BALL_WEIGHTS, YOLO_BALL_WEIGHTS_WHITE, YOLO_BALL_WEIGHTS_RED
from models.yolo_detect import YOLOBallDetector
# NOTE: Ball detection implementation was switched from Ultralytics YOLO to
# Roboflow RF-DETR on 2025-11-16. The model wrapper exposed as
# `models.yolo_detect.YOLOBallDetector` maintains the same `detect(img)`
# interface and returns a list of (x,y,w,h,confidence). Old YOLO-based
# implementation is preserved commented-out in `models/yolo_detect.py`.
from detection.ball_tracker import ball_detect
from cvzone.ColorModule import ColorFinder

_yolo_detector = None
_selected_ball_color = None
_selected_weights_path = None


def select_ball_color_interactive():
    """Ask the user which ball color to detect and return the corresponding weights path.

    The selection is cached for subsequent calls.
    """
    global _selected_ball_color, _selected_weights_path
    if _selected_ball_color is not None and _selected_weights_path is not None:
        return _selected_ball_color, _selected_weights_path

    valid = {'white': YOLO_BALL_WEIGHTS_WHITE, 'red': YOLO_BALL_WEIGHTS_RED}
    prompt = "Select ball color for detection ('white' or 'red') [white]: "
    try:
        choice = input(prompt).strip().lower() or 'white'
    except Exception:
        # If input is not available (non-interactive), default to white
        choice = 'white'

    if choice not in valid:
        print(f"Unknown choice '{choice}', defaulting to 'white'.")
        choice = 'white'

    weights_path = valid.get(choice)
    # Fallback to generic weight if specific file not present
    if not os.path.exists(weights_path):
        print(f"Warning: weights for '{choice}' not found at {weights_path}. Falling back to default ball weights.")
        weights_path = YOLO_BALL_WEIGHTS

    _selected_ball_color = choice
    _selected_weights_path = weights_path
    print(f"Using '{_selected_ball_color}' ball model: {_selected_weights_path}")
    return _selected_ball_color, _selected_weights_path


def get_yolo_detector(weights_path=None):
    """Return a YOLO detector instance. If weights_path is None, use the
    selected per-color weights (interactive prompt) or the default.
    """
    global _yolo_detector
    if weights_path is None:
        # If not specified, interactively choose color and map to weights
        _, weights_path = select_ball_color_interactive()

    if _yolo_detector is None:
        _yolo_detector = YOLOBallDetector(weights_path)
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
        # Use overlay blending controlled by detector VIS_OPACITY if present
        alpha = getattr(detector, 'VIS_OPACITY', 0.75)
        for (x, y, w, h, confidence) in filtered:
            # Draw filled rectangle on overlay then blend
            overlay = frame_with_ball.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, frame_with_ball, 1 - alpha, 0, frame_with_ball)
            # Draw border and label on top
            cv2.rectangle(frame_with_ball, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_with_ball, f"Ball {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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

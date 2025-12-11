import cv2
from yolo_detect import YOLOBallDetector
from ball_tracker import ball_detect
from cvzone.ColorModule import ColorFinder

# Initialize detectors once
yolo_detector = YOLOBallDetector('weights/yolov8_cricket_ball2/weights/best.pt')  # or your custom weights
color_finder = ColorFinder(False)
hsv_vals = {
    "hmin": 10, "smin": 44, "vmin": 192,
    "hmax": 125, "smax": 114, "vmax": 255,
}

def detect_ball_on_frame(frame):
    """
    Draw ball detection (YOLO or color-based) on top of the input frame.
    Returns (frame_with_ball, detected:bool)
    """
    frame_with_ball = frame.copy()
    found = False

    # Try YOLO detection with size/aspect filtering
    yolo_detections = yolo_detector.detect(frame)
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
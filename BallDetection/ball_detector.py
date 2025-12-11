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
    Detect ball using YOLO, falling back to Color detection.
    
    Returns:
        frame: The UNTOUCHED original frame.
        ball_data: None if no ball found, otherwise a list:
                   [center_x, center_y, radius, confidence, label]
    """
    ball_data = None

    # 1. Try YOLO detection
    yolo_detections = yolo_detector.detect(frame)
    filtered = []
    
    for (x, y, w, h, confidence) in yolo_detections:
        aspect = w / h if h > 0 else 0
        area = w * h
        # Size/Aspect filtering
        if 0.7 < aspect < 1.3 and 100 < area < 3000:
            filtered.append((x, y, w, h, confidence))
            
    if filtered:
        # Take the highest confidence detection
        best_det = max(filtered, key=lambda d: d[4])
        x, y, w, h, conf = best_det
        
        # Convert Top-Left (x,y) to Center (cx, cy)
        cx = x + w // 2
        cy = y + h // 2
        radius = max(w, h) // 2
        
        ball_data = [cx, cy, radius, conf, "Ball (YOLO)"]

    # 2. Fallback to color-based detection
    else:
        # ball_detect now returns (img, x, y) without drawing
        _, bx, by = ball_detect(frame, color_finder, hsv_vals)
        
        if bx != 0 and by != 0:
            # Color detector returns center x, y. 
            # We assume a fixed radius since the helper doesn't return area/radius.
            radius = 10 
            ball_data = [bx, by, radius, 1.0, "Ball (Color)"]

    return frame, ball_data
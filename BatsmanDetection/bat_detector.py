from ultralytics import YOLO
import cv2

# Load your trained bat detector
# 


bat_model = YOLO('weights/Bat_detection2/weights/best.pt')
#bat_model = YOLO('weights/bat_weights_new.pt')

def detect_bat(frame, conf=0.25, center_fraction=0.25): 
    """
    Run the bat detection model on `frame`, filtering for detections
    in the horizontal center of the frame.
    
    Args:
        frame: The input image/frame
        conf (float): Confidence threshold
        center_fraction (float): The percentage of the width to scan in the center (0.0 to 1.0).
                                 0.5 means checking the middle 50% of the frame.
    
    Returns (frame, detections_list)
    """
    results = bat_model.predict(frame, conf=conf, verbose=False)
    detections = []

    # 1. Get frame dimensions to calculate the center zone
    height, width = frame.shape[:2]
    
    # 2. Calculate the horizontal boundaries (ROI)
    # If width is 1920 and center_fraction is 0.5, we want the middle 960 pixels.
    zone_width = width * center_fraction
    x_min = (width - zone_width) / 2
    x_max = (width + zone_width) / 2

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            
            # 3. Calculate the center of the detection box
            det_center_x = (x1 + x2) / 2
            
            # 4. Filter: Only keep detections where the center is inside our horizontal zone
            if x_min < det_center_x < x_max:
                conf_score = float(box.conf[0])
                label = "Bat"

                detections.append({
                    "label": label,
                    "conf": round(conf_score, 4),
                    "box": [x1, y1, x2 - x1, y2 - y1] # x, y, w, h
                })
    return frame, detections
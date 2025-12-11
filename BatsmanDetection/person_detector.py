# file: person_detector.py
from ultralytics import YOLO
import cv2

# Load general person detector (COCO)
person_model = YOLO('weights/yolov8n.pt')  # adjust model file if needed

def detect_persons(frame, person_conf: float = 0.5, center_ratio: float = 0.25):
    """
    Detect only persons in the central vertical strip of the frame.
    
    Args:
        frame: The input image.
        person_conf (float): Confidence threshold for YOLO.
        center_ratio (float): The fraction of screen width to consider "center".
                              0.5 means the middle 50% of the width.
                              
    Returns:
        frame: The original frame (unmodified).
        detections: list of tuples (x, y, w, h, "Person")
    """
    h_img, w_img = frame.shape[:2]
    
    # Define the horizontal boundaries for the "center" region
    # If center_ratio is 0.5, we want the middle 50%.
    # Margin on each side is (1 - 0.5) / 2 = 0.25
    margin = (1.0 - center_ratio) / 2.0
    x_min_boundary = int(w_img * margin)
    x_max_boundary = int(w_img * (1.0 - margin))

    # Run inference on the frame
    results = person_model.predict(frame, conf=person_conf, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = person_model.names.get(cls_id, str(cls_id))
            
            # Filter only for 'person' class
            if label != "person":
                continue
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Calculate the center X of the detection box
            center_x = (x1 + x2) // 2
            
            # Check if the center X is within the central boundaries
            if x_min_boundary <= center_x <= x_max_boundary:
                # Append to list: (x, y, width, height, Label)
                detections.append((x1, y1, x2 - x1, y2 - y1, "Person"))

    # Return the frame unmodified + the list of detection data
    return frame, detections
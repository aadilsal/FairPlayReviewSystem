from ultralytics import YOLO
import cv2

# Load general person detector (COCO)
person_model = YOLO('weights/yolov8n.pt')  # adjust model file if needed

def detect_persons(frame, person_conf: float = 0.5):
    """
    Detect only persons in the frame using a COCO pretrained YOLO model.
    Returns (frame, detections)
    detections: list of tuples (x, y, w, h, "Person")
    """
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
            
            # Append to list: (x, y, width, height, Label)
            detections.append((x1, y1, x2 - x1, y2 - y1, "Person"))

    # Return the frame unmodified + the list of detection data
    return frame, detections
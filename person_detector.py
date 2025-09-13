# person_detector.py
import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model (trained on COCO, includes "person")
model = YOLO("yolov8s.pt")  # or yolov8n.pt for faster but less accurate

def detect_persons(frame, conf_threshold=0.5):
    """
    Detect persons in a frame using YOLOv8 pretrained on COCO dataset.
    Returns list of (x1, y1, x2, y2, confidence) for each detected person.
    """
    results = model.predict(frame, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "person":  # only keep person detections
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    return frame, detections

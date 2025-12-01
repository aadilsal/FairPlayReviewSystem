from ultralytics import YOLO
import cv2

# Load general person detector (COCO)
person_model = YOLO('yolov8n.pt')  # adjust model file if needed

def detect_persons(frame, person_conf: float = 0.5):
    """
    Detect only persons in the frame using a COCO pretrained YOLO model.
    Returns (annotated_frame, detections)
    detections: list of tuples (x, y, w, h, "Person")
    """
    results = person_model.predict(frame, conf=person_conf, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = person_model.names.get(cls_id, str(cls_id))
            if label != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            detections.append((x1, y1, x2 - x1, y2 - y1, "Person"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf_score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detections
from ultralytics import YOLO
import cv2
import os
from config.paths import BATSMAN_WEIGHTS


# Helper to safely load a YOLO model path with a fallback
def _load_yolo_model(path, fallback='yolov8n.pt'):
    if path and os.path.exists(path):
        try:
            print(f"[INFO] Loading YOLO model from {path}")
            return YOLO(path)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    # fallback
    print(f"[INFO] Falling back to {fallback}")
    return YOLO(fallback)


# Load your trained batsman detector (can replace path after training)
batsman_model = _load_yolo_model(BATSMAN_WEIGHTS)
# Load general person detector (COCO) - use a small model by default for speed
person_model = _load_yolo_model('yolov8s.pt')  # or yolov8n.pt for faster inference


def set_batsman_weights(path: str):
    """Reload the batsman detector with new weights at runtime.

    Call this after training completes so the running pipeline can use the
    newly trained batsman detector without restarting the process.
    """
    global batsman_model
    batsman_model = _load_yolo_model(path)
    return batsman_model

def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def detect_persons(frame, batsman_conf=0.3, person_conf=0.6, iou_threshold=0.5, batsman_model_override=None, person_model_override=None):
    # Allow callers to override models at runtime (useful for integration/testing)
    bm = batsman_model_override if batsman_model_override is not None else batsman_model
    pm = person_model_override if person_model_override is not None else person_model

    batsman_results = bm.predict(frame, conf=batsman_conf, verbose=False)
    batsman_boxes = []
    detections = []

    # Detect batsman (blue box)
    for result in batsman_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            batsman_boxes.append((x1, y1, x2, y2))
            detections.append((x1, y1, x2 - x1, y2 - y1, "Batsman"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Batsman", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Detect general persons (green box)
    person_results = pm.predict(frame, conf=person_conf, verbose=False)
    for result in person_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = person_model.names[cls_id]
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Check overlap with batsman boxes
                overlap = False
                for b_box in batsman_boxes:
                    if iou((x1, y1, x2, y2), b_box) > iou_threshold:
                        overlap = True
                        break
                if not overlap:
                    detections.append((x1, y1, x2 - x1, y2 - y1, "Person"))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detections

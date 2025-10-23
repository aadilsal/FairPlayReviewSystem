from ultralytics import YOLO
import cv2

# Load your trained batsman detector
batsman_model = YOLO('outputs/batsman_detection/weights/best.pt')
# Load general person detector (COCO)
person_model = YOLO('yolov8s.pt')  # or yolov8n.pt for faster inference

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

def detect_persons(frame, batsman_conf=0.3, person_conf=0.6, iou_threshold=0.5):
    batsman_results = batsman_model.predict(frame, conf=batsman_conf, verbose=False)
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
    person_results = person_model.predict(frame, conf=person_conf, verbose=False)
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

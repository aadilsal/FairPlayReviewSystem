from ultralytics import YOLO
import cv2

# Load your trained bat detector
bat_model = YOLO('outputs/Bat_detection2/weights/best.pt')  # Adjust path if necessary

def detect_bat(frame, conf= 0.25):  # Ensure bat_conf is a parameter
    """
    Run the bat detection model on `frame`.
    Returns (frame_with_drawings, detections_list)
    detections_list: list of dicts { 'label': str, 'conf': float, 'box': [x, y, w, h] }
    """
    results = bat_model.predict(frame, conf=conf, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            conf_score = float(box.conf[0])
            label = "Bat"

            detections.append({
                "label": label,
                "conf": round(conf_score, 4),
                "box": [x1, y1, x2 - x1, y2 - y1]
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for bat
            cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, detections
from ultralytics import YOLO
import cv2

# Load your trained bat detector
bat_model = YOLO('weights/Bat_detection2/weights/best.pt')  

def detect_bat(frame, conf=0.25):  # Ensure bat_conf is a parameter
    """
    Run the bat detection model on `frame`.
    Returns (frame_untouched, detections_list)
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

    # Return the frame unmodified + the list of detection data
    return frame, detections
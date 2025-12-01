import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# Path to your new players model weights (adjust if needed)
MODEL_PATH = Path("outputs/players_detection") / "weights" / "best.pt"

# Load model once at import
try:
    player_model = YOLO(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load players model at {MODEL_PATH}: {e}")

# Color map for classes (BGR)
CLASS_COLORS = {
    "Batsman": (255, 0, 0),        # blue
    "Bowler": (0, 255, 0),         # green
    "Umpire": (0, 255, 255),       # yellow
    "Wicket Keeper": (255, 255, 0),# light cyan
    "players": (255, 0, 255),      # magenta (fallback)
}

def detect_players(frame, conf: float = 0.5):
    """
    Run the players detection model on `frame`.
    Returns (frame_with_drawings, detections_list)
    detections_list: list of dicts { 'label': str, 'conf': float, 'box': [x, y, w, h] }
    """
    results = player_model.predict(frame, conf=conf, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            # xyxy, conf and cls are tensors/arrays; convert to Python types
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            w, h = x2 - x1, y2 - y1
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            # obtain class name from model (fallback to cls_id if missing)
            label = player_model.names.get(cls_id, str(cls_id))

            detections.append({
                "label": label,
                "conf": round(conf_score, 4),
                "box": [x1, y1, w, h]
            })

            color = CLASS_COLORS.get(label, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, detections
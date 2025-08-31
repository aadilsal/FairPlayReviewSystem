import kagglehub
import os
from ultralytics import YOLO
import cv2

# -------------------------------
# 1. Download Dataset from Kaggle
# -------------------------------
# Cricket Ball Dataset for YOLO: https://www.kaggle.com/datasets/kushagra3204/cricket-ball-dataset-for-yolo
dataset_path = kagglehub.dataset_download("kushagra3204/cricket-ball-dataset-for-yolo")
print("Dataset downloaded at:", dataset_path)

# -------------------------------
# 2. Train YOLOv8 Model
# -------------------------------
# Load YOLOv8 model (pretrained on COCO for transfer learning)
model = YOLO("yolov8n.pt")  # use yolov8n (nano) for speed or yolov8s/m/l for accuracy

# Train the model
model.train(
    data="Cricket Ball Detection.v1i.yolov8/data.yaml",  # Use local dataset YAML file
    epochs=30,
    imgsz=640,
    batch=16
)


# Save best model
model_path = "cricket_ball_detector.pt"
model.save(model_path)
print(f"Model saved as {model_path}")

# -------------------------------
# 3. Run Inference on Video / Image
# -------------------------------
# Replace with path to test video or image
test_video = os.path.join(dataset_path, "test", "video.mp4")  # example
cap = cv2.VideoCapture(test_video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Cricket Ball Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

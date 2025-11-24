"""Register batsman and ball YOLO models to MLflow/DagsHub."""
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import shutil
import tempfile
import json

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("cricket-yolo-detection")

MODELS = [
    {
        "path": Path("weights/yolov8s-pose.pt"),
        "name": "yolo-batsman-detector",
        "type": "batsman",
        "description": "YOLOv8s-pose batsman detector"
    },
    {
        "path": Path("backup_ball_detection/weights/ball_detector_best_20251118_235310.pt"),
        "name": "yolo-ball-detector",
        "type": "ball",
        "description": "YOLO ball detector"
    }
]

def save_yolo_model_to_mlflow(model_path: Path, model_name: str, model_type: str):
    if not model_path.exists():
        print(f"[ERROR] Model weights not found: {model_path}")
        return None
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", "YOLO Object Detection")
        mlflow.log_param("framework", "pytorch")
        mlflow.log_param("detector_source", "Ultralytics YOLO")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("detection_target", model_type)
        mlflow.log_param("weights_file", model_path.name)
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        mlflow.log_metric("model_size_mb", round(file_size_mb, 2))
        temp_dir = tempfile.mkdtemp()
        try:
            model_artifact_path = Path(temp_dir) / "weights"
            model_artifact_path.mkdir(exist_ok=True)
            dest_path = model_artifact_path / "best.pt"
            shutil.copy2(model_path, dest_path)
            metadata = {
                "model_name": model_name,
                "format": "pytorch_yolo",
                "framework": "ultralytics",
                "model_type": model_type,
                "weights_file": "best.pt"
            }
            metadata_path = Path(temp_dir) / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifacts(str(model_artifact_path), artifact_path="weights")
            mlflow.log_artifact(str(metadata_path))
            print(f"[SUCCESS] Model '{model_name}' logged to MLflow (Run ID: {run_id})")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return run_id

def main():
    print("="*60)
    print("Registering batsman and ball YOLO models to MLflow/DagsHub")
    print("="*60)
    for model in MODELS:
        print(f"\nRegistering: {model['description']}")
        run_id = save_yolo_model_to_mlflow(model["path"], model["name"], model["type"])
        if run_id:
            print(f"Run ID: {run_id}")
            print(f"View in MLflow: {os.getenv('MLFLOW_TRACKING_URI')}")
        else:
            print(f"[ERROR] Failed to register {model['name']}")
    print("\nRegistration complete.")

if __name__ == "__main__":
    main()

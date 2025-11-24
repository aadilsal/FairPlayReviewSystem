"""Train/register YOLO model to MLflow for cricket detection."""
import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import json

load_dotenv()

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("cricket-yolo-detection")


def save_yolo_model_to_mlflow(model_path: str, model_name: str, model_type: str = "ball"):
    """
    Save YOLO model weights to MLflow as artifacts.
    
    Args:
        model_path: Path to YOLO .pt weights file
        model_name: Name for the model in MLflow
        model_type: Type of model (ball, batsman, person)
    
    Returns:
        MLflow run_id
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log model information as parameters
        mlflow.log_param("model_type", "YOLO Object Detection")
        mlflow.log_param("framework", "pytorch")
        mlflow.log_param("detector_source", "Ultralytics YOLO")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("detection_target", model_type)
        mlflow.log_param("weights_file", os.path.basename(model_path))
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        mlflow.log_metric("model_size_mb", round(file_size_mb, 2))
        
        # Create temporary directory for artifacts
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Copy model weights to temp directory
            model_artifact_path = Path(temp_dir) / "weights"
            model_artifact_path.mkdir(exist_ok=True)
            
            dest_path = model_artifact_path / "best.pt"
            shutil.copy2(model_path, dest_path)
            print(f"✓ Copied model weights to: {dest_path}")
            
            # Create metadata file
            metadata = {
                "model_name": model_name,
                "format": "pytorch_yolo",
                "framework": "ultralytics",
                "model_type": model_type,
                "weights_file": "best.pt",
                "usage": {
                    "load": "from ultralytics import YOLO; model = YOLO('best.pt')",
                    "inference": "results = model(image, conf=0.25)"
                }
            }
            
            metadata_path = Path(temp_dir) / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Log artifacts to MLflow
            print("Uploading model artifacts to MLflow...")
            mlflow.log_artifacts(str(model_artifact_path), artifact_path="weights")
            mlflow.log_artifact(str(metadata_path))
            
            print(f"✓ Model logged to MLflow (Run ID: {run_id})")
            print(f"✓ Model artifacts uploaded successfully")
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return run_id


def register_existing_weights():
    """Register existing YOLO weights from the project."""
    print("=" * 60)
    print("Registering YOLO Models to MLflow")
    print("=" * 60)
    
    base_path = Path(__file__).resolve().parents[1]
    
    # Define available models
    models = [
        {
            "path": base_path / "yolov8n.pt",
            "name": "yolo-cricket-general-v8n",
            "type": "general",
            "description": "YOLOv8n - General purpose detector"
        },
        {
            "path": base_path / "yolov8m.pt",
            "name": "yolo-cricket-general-v8m",
            "type": "general",
            "description": "YOLOv8m - Medium model for better accuracy"
        },
        {
            "path": base_path / "yolo11n.pt",
            "name": "yolo-cricket-general-v11n",
            "type": "general",
            "description": "YOLO11n - Latest nano model"
        }
    ]
    
    # Check for custom trained weights
    weights_dir = base_path / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            models.append({
                "path": weight_file,
                "name": f"yolo-cricket-{weight_file.stem}",
                "type": "custom",
                "description": f"Custom trained: {weight_file.name}"
            })
    
    # Check backup ball detection weights
    backup_weights = base_path / "backup_ball_detection" / "weights"
    if backup_weights.exists():
        for weight_file in backup_weights.glob("*.pt"):
            models.append({
                "path": weight_file,
                "name": f"yolo-ball-{weight_file.stem}",
                "type": "ball",
                "description": f"Ball detector: {weight_file.name}"
            })
    
    registered_models = []
    
    print("\nAvailable models to register:")
    for i, model in enumerate(models, 1):
        exists = "✓" if model["path"].exists() else "✗"
        print(f"{i}. [{exists}] {model['description']}")
        print(f"   Path: {model['path']}")
    
    print("\n" + "=" * 60)
    print("Registering models...")
    print("=" * 60)
    
    for model in models:
        if not model["path"].exists():
            print(f"\n✗ Skipping {model['name']} (file not found)")
            continue
        
        try:
            print(f"\n📦 Registering: {model['description']}")
            run_id = save_yolo_model_to_mlflow(
                str(model["path"]),
                model["name"],
                model["type"]
            )
            registered_models.append({
                "name": model["name"],
                "run_id": run_id,
                "type": model["type"],
                "path": str(model["path"])
            })
            print(f"✓ Registered with Run ID: {run_id}")
        except Exception as e:
            print(f"✗ Failed to register {model['name']}: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Registration Complete!")
    print("=" * 60)
    
    if registered_models:
        print("\nRegistered Models:")
        for model in registered_models:
            print(f"\n  • {model['name']}")
            print(f"    Type: {model['type']}")
            print(f"    Run ID: {model['run_id']}")
            print(f"    Source: {model['path']}")
        
        # Suggest default model
        default_model = registered_models[0]
        print("\n" + "=" * 60)
        print("Recommended Configuration:")
        print("=" * 60)
        print(f"\nUpdate your .env file:")
        print(f"MODEL_RUN_ID={default_model['run_id']}")
        print(f"\nThis will use: {default_model['name']}")
        print(f"\nView in MLflow:")
        print(f"→ {os.getenv('MLFLOW_TRACKING_URI')}")
    else:
        print("\n⚠️  No models were registered. Please check your weights files.")
    
    return registered_models


def main():
    import sys
    
    if len(sys.argv) > 1:
        # Register specific model
        model_path = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else "yolo-cricket-detector"
        model_type = sys.argv[3] if len(sys.argv) > 3 else "general"
        
        print(f"Registering model: {model_path}")
        run_id = save_yolo_model_to_mlflow(model_path, model_name, model_type)
        print(f"\n✓ Model registered with Run ID: {run_id}")
        print(f"\nUpdate .env with: MODEL_RUN_ID={run_id}")
    else:
        # Register all available models
        register_existing_weights()


if __name__ == "__main__":
    main()

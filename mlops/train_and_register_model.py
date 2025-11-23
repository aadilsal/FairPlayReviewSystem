"""Train/load TensorFlow object detection model and register to MLflow."""
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import tensorflow as tf
import tensorflow_hub as hub

load_dotenv()

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("object-detection")


def create_detection_model():
    """
    Load pre-trained TensorFlow object detection model from TensorFlow Hub
    """
    # Using EfficientDet Lite2 (fast and efficient for video inference)
    detector_url = "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1"
    
    print(f"Loading model from: {detector_url}")
    detector = hub.load(detector_url)
    
    return detector


def save_model_to_mlflow_basic(model, model_name):
    """
    Save TensorFlow model to MLflow using basic artifact logging (DagsHub compatible)
    """
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log model information as parameters
        mlflow.log_param("model_type", "TensorFlow Object Detection")
        mlflow.log_param("framework", "tensorflow")
        mlflow.log_param("detector_source", "TensorFlow Hub - EfficientDet Lite2")
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("input_size", 320)
        
        # Save model to temporary directory
        temp_dir = tempfile.mkdtemp()
        model_path = Path(temp_dir) / "saved_model"
        
        try:
            # Save as TensorFlow SavedModel format
            print(f"Saving model to temporary directory: {model_path}")
            tf.saved_model.save(model, str(model_path))
            
            # Log the entire saved model directory as artifacts
            print("Uploading model artifacts to MLflow...")
            mlflow.log_artifacts(str(model_path), artifact_path="model")
            
            # Log a metadata file
            metadata = {
                "model_name": model_name,
                "format": "tensorflow_savedmodel",
                "source": "TensorFlow Hub EfficientDet Lite2"
            }
            import json
            metadata_path = Path(temp_dir) / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_path))
            
            print(f"✓ Model logged to MLflow (Run ID: {run_id})")
            print(f"✓ Model artifacts uploaded successfully")
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return run_id


def main():
    model_name = os.getenv("MLFLOW_MODEL_NAME", "tf-object-detection")
    
    print("=" * 60)
    print("Training & Registering TensorFlow Object Detection Model")
    print("=" * 60)
    
    # Create model
    print("\n1. Loading pre-trained TensorFlow model...")
    model = create_detection_model()
    print("✓ Model loaded successfully")
    
    # Save to MLflow
    print("\n2. Saving model to MLflow...")
    run_id = save_model_to_mlflow_basic(
        model,
        model_name
    )
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print(f"Model Name: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"\nModel artifacts saved in MLflow run")
    print(f"Load using: runs:/{run_id}/model")
    print(f"\nView in MLflow:")
    print(f"→ {os.getenv('MLFLOW_TRACKING_URI')}")
    print("\nTo use this model:")
    print(f"1. Update MODEL_RUN_ID={run_id} in .env")
    print("2. Model will be loaded from artifacts in inference pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()

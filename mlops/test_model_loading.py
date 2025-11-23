"""Test loading model from MLflow run artifacts."""
import os
from dotenv import load_dotenv
import mlflow
import tensorflow as tf
import numpy as np

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model_run_id = os.getenv("MODEL_RUN_ID")

if not model_run_id:
    print("✗ MODEL_RUN_ID not set in .env")
    print("Please run mlops/train_and_register_model.py first")
    exit(1)

print(f"Testing model loading from MLflow artifacts...")
print(f"Run ID: {model_run_id}")
print(f"Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")

try:
    # Load model from artifacts
    artifact_uri = f"runs:/{model_run_id}/model"
    print(f"\nDownloading from: {artifact_uri}")
    
    local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    print(f"Downloaded to: {local_path}")
    
    # Load TensorFlow model
    print("Loading TensorFlow SavedModel...")
    model = tf.saved_model.load(local_path)
    
    print("✓ Model loaded successfully from MLflow artifacts!")
    print(f"Model type: {type(model)}")
    
    # Test inference with dummy data
    print("\nTesting inference with dummy image...")
    dummy_image = np.random.randint(0, 255, (1, 320, 320, 3), dtype=np.uint8)
    dummy_tensor = tf.convert_to_tensor(dummy_image)
    
    detections = model(dummy_tensor)
    
    print("✓ Inference test successful!")
    print(f"Output type: {type(detections)}")
    
    # Handle both dict and tuple outputs
    if isinstance(detections, dict):
        print(f"Detection keys: {list(detections.keys())}")
        if 'num_detections' in detections:
            print(f"Number of detections: {detections['num_detections'][0].numpy()}")
    elif isinstance(detections, tuple):
        print(f"Output tuple length: {len(detections)}")
    
    print("\n✓ Model is ready for use in the inference pipeline")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    raise

"""Debug script to check TensorFlow Hub model output format."""
import os
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from model_manager import ModelManager
from config import load_config

load_dotenv()

def main():
    print("=" * 60)
    print("Debugging TensorFlow Hub Model Output Format")
    print("=" * 60)
    
    cfg = load_config()
    mm = ModelManager(cfg.mlflow_tracking_uri, cfg.mlflow_username, cfg.mlflow_password)
    
    print(f"\nLoading model from run: {cfg.model_run_id}")
    model = mm.get_model(cfg.model_run_id)
    print("✓ Model loaded\n")
    
    # Create a dummy input image (320x320x3 for EfficientDet Lite2)
    print("Creating test input (320x320x3 RGB image)...")
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    input_tensor = tf.convert_to_tensor(dummy_img, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input dtype: {input_tensor.dtype}\n")
    
    # Run inference
    print("Running inference...")
    detections = model(input_tensor)
    
    print("=" * 60)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    
    # Analyze output type
    print(f"\n1. Output Type: {type(detections)}")
    print(f"   Type name: {type(detections).__name__}")
    
    # Check if it's a dict-like object
    if hasattr(detections, 'keys'):
        print(f"\n2. Has 'keys' attribute: Yes")
        try:
            keys = list(detections.keys())
            print(f"   Available keys: {keys}")
            
            print(f"\n3. Analyzing each output:")
            for key in keys:
                value = detections[key]
                print(f"\n   Key: '{key}'")
                print(f"   - Type: {type(value)}")
                print(f"   - Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                print(f"   - Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
                if hasattr(value, 'numpy'):
                    print(f"   - Has numpy(): Yes")
                    arr = value.numpy()
                    print(f"   - Numpy shape: {arr.shape}")
                    print(f"   - Sample value: {arr.flatten()[:3]}")
        except Exception as e:
            print(f"   Error accessing keys: {e}")
    
    # Check if it's a tuple
    elif isinstance(detections, tuple):
        print(f"\n2. Is tuple: Yes")
        print(f"   Tuple length: {len(detections)}")
        print(f"\n3. Analyzing each element:")
        for i, item in enumerate(detections):
            print(f"\n   Index {i}:")
            print(f"   - Type: {type(item)}")
            print(f"   - Shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
            print(f"   - Dtype: {item.dtype if hasattr(item, 'dtype') else 'N/A'}")
    
    # Check if it's a list
    elif isinstance(detections, list):
        print(f"\n2. Is list: Yes")
        print(f"   List length: {len(detections)}")
    
    # Other type
    else:
        print(f"\n2. Special TensorFlow object")
        print(f"   Dir: {[attr for attr in dir(detections) if not attr.startswith('_')][:20]}")
        
        # Try common TF Hub patterns
        if hasattr(detections, 'numpy'):
            print(f"   Has numpy() method")
            try:
                arr = detections.numpy()
                print(f"   Numpy shape: {arr.shape}")
            except Exception as e:
                print(f"   Error calling numpy(): {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Quick test to verify preprocessing integration in the pipeline.
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Create a test frame with poor quality
def create_test_frame():
    """Create a synthetic dark, blurry frame for testing."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark frame
    
    # Add some features (simulated balls)
    cv2.circle(frame, (200, 200), 20, (100, 100, 100), -1)
    cv2.circle(frame, (400, 300), 18, (100, 100, 100), -1)
    
    # Add some blur
    frame = cv2.GaussianBlur(frame, (7, 7), 2.0)
    
    return frame

def test_pipeline_integration():
    """Test preprocessing integration in pipeline."""
    print("\n" + "="*70)
    print("TESTING PREPROCESSING PIPELINE INTEGRATION")
    print("="*70)
    
    # Create test directory and frame
    test_dir = "outputs/test_preprocessing_pipeline"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test frames
    print("\n📸 Creating test frames...")
    for i in range(3):
        frame = create_test_frame()
        frame_path = os.path.join(test_dir, f"test_frame_{i:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"   Created: {frame_path}")
    
    # Test with pipeline
    print("\n🔄 Testing pipeline with preprocessing...")
    from pipeline.main_pipeline import process_frames_pipeline
    
    frame_paths = sorted(Path(test_dir).glob("test_frame_*.jpg"))
    frame_paths = [str(p) for p in frame_paths]
    
    try:
        process_frames_pipeline(
            frame_paths,
            enable_motion_prediction=False,  # Disable for quick test
            enable_preprocessing=True,       # Enable preprocessing
            target_brightness=0.5
        )
        print("\n✅ Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    return True

if __name__ == "__main__":
    success = test_pipeline_integration()
    sys.exit(0 if success else 1)

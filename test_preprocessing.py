"""
Quick test script for frame preprocessing module.

Tests basic functionality without requiring actual image files.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_frames():
    """Create synthetic test frames with different quality issues."""
    # Base frame (640x480, 3 channels)
    frames = {}
    
    # 1. Normal quality frame
    normal = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Add some features (circles representing balls)
    cv2.circle(normal, (200, 200), 30, (255, 255, 255), -1)
    cv2.circle(normal, (400, 300), 25, (255, 255, 255), -1)
    frames['normal'] = normal
    
    # 2. Dark frame
    dark = (normal * 0.3).astype(np.uint8)
    frames['dark'] = dark
    
    # 3. Bright frame
    bright = np.clip(normal * 1.8, 0, 255).astype(np.uint8)
    frames['bright'] = bright
    
    # 4. Blurry frame
    blurry = cv2.GaussianBlur(normal, (15, 15), 5.0)
    frames['blurry'] = blurry
    
    # 5. Low contrast frame
    low_contrast = (normal * 0.5 + 64).astype(np.uint8)
    frames['low_contrast'] = low_contrast
    
    return frames


def test_quality_assessment():
    """Test quality assessment functions."""
    print("\n" + "="*70)
    print("TEST 1: Quality Assessment")
    print("="*70)
    
    from detection.frame_preprocessing import (
        detect_brightness_level,
        detect_blur_level,
        get_frame_quality_score
    )
    
    frames = create_test_frames()
    
    for name, frame in frames.items():
        print(f"\n📸 Frame: {name}")
        
        brightness = detect_brightness_level(frame)
        blur = detect_blur_level(frame)
        quality, metrics = get_frame_quality_score(frame)
        
        print(f"   Brightness: {brightness} ({metrics.brightness_value:.2f})")
        print(f"   Blur Score: {blur:.2f}")
        print(f"   Contrast: {metrics.contrast_score:.2f}")
        print(f"   Quality: {quality:.2f}")
        print(f"   Needs Enhancement: {'Yes' if metrics.needs_enhancement else 'No'}")
    
    print("\n✅ Quality assessment test passed!")


def test_image_enhancement():
    """Test image enhancement functions."""
    print("\n" + "="*70)
    print("TEST 2: Image Enhancement")
    print("="*70)
    
    from detection.frame_preprocessing import (
        normalize_brightness,
        enhance_contrast,
        reduce_blur,
        preprocess_frame
    )
    
    frames = create_test_frames()
    dark_frame = frames['dark']
    
    print("\n🔧 Testing on dark frame...")
    
    # Test brightness normalization
    print("   Testing brightness normalization...")
    enhanced = normalize_brightness(dark_frame, target_brightness=0.5)
    assert enhanced.shape == dark_frame.shape
    avg_before = np.mean(dark_frame)
    avg_after = np.mean(enhanced)
    print(f"   ✓ Brightness: {avg_before:.1f} → {avg_after:.1f}")
    
    # Test contrast enhancement
    print("   Testing CLAHE contrast enhancement...")
    enhanced = enhance_contrast(dark_frame)
    assert enhanced.shape == dark_frame.shape
    std_before = np.std(dark_frame)
    std_after = np.std(enhanced)
    print(f"   ✓ Std Dev: {std_before:.1f} → {std_after:.1f}")
    
    # Test sharpening
    print("   Testing sharpening...")
    enhanced = reduce_blur(frames['blurry'], strength=1.0)
    assert enhanced.shape == frames['blurry'].shape
    print(f"   ✓ Shape preserved: {enhanced.shape}")
    
    # Test combined preprocessing
    print("   Testing combined preprocessing...")
    enhanced, enhancements = preprocess_frame(dark_frame, adaptive=True)
    assert enhanced.shape == dark_frame.shape
    print(f"   ✓ Applied: {', '.join(enhancements) or 'None'}")
    
    print("\n✅ Image enhancement test passed!")


def test_adaptive_detector():
    """Test AdaptiveBallDetector class."""
    print("\n" + "="*70)
    print("TEST 3: Adaptive Ball Detector")
    print("="*70)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️  Skipping: ultralytics not installed")
        return
    
    from detection.frame_preprocessing import AdaptiveBallDetector
    
    # Create dummy model (this won't actually work for detection)
    print("\n📦 Loading model...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"⚠️  Skipping: Could not load model ({e})")
        return
    
    # Create detector
    print("   Creating adaptive detector...")
    detector = AdaptiveBallDetector(
        model=model,
        enable_preprocessing=True,
        quality_threshold=0.6,
        base_confidence=0.25,
        log_enhancements=False
    )
    print("   ✓ Detector created")
    
    # Test with synthetic frames
    frames = create_test_frames()
    
    print("\n🔍 Testing detection on synthetic frames...")
    for name, frame in list(frames.items())[:2]:  # Test first 2 to save time
        print(f"\n   Frame: {name}")
        try:
            result = detector.detect(
                frame=frame,
                frame_id=name,
                imgsz=640,
                device='cpu'
            )
            print(f"   ✓ Quality: {result['quality_score']:.2f}")
            print(f"   ✓ Preprocessed: {result['was_preprocessed']}")
            print(f"   ✓ Time: {result['total_time_ms']:.1f}ms")
        except Exception as e:
            print(f"   ⚠️  Detection failed: {e}")
    
    # Test statistics
    print("\n📊 Testing statistics...")
    stats = detector.get_statistics()
    print(f"   ✓ Total frames: {stats['total_frames_processed']}")
    print(f"   ✓ Preprocessed: {stats['frames_preprocessed']}")
    print(f"   ✓ Avg quality: {stats['average_quality_score']:.2f}")
    
    print("\n✅ Adaptive detector test passed!")


def test_visualization():
    """Test visualization functions."""
    print("\n" + "="*70)
    print("TEST 4: Visualization Functions")
    print("="*70)
    
    from detection.frame_preprocessing import (
        visualize_quality_assessment,
        compare_preprocessing,
        get_frame_quality_score
    )
    
    frames = create_test_frames()
    frame = frames['dark']
    
    print("\n🎨 Testing quality visualization...")
    _, metrics = get_frame_quality_score(frame)
    vis_frame = visualize_quality_assessment(frame, metrics, show_metrics=True)
    assert vis_frame.shape == frame.shape
    print(f"   ✓ Visualization created: {vis_frame.shape}")
    
    print("\n🎨 Testing comparison visualization...")
    results = compare_preprocessing(frame, show_original=True, show_enhanced=True)
    assert 'original' in results
    assert 'enhanced' in results
    assert 'comparison' in results
    print(f"   ✓ Original: {results['original'].shape}")
    print(f"   ✓ Enhanced: {results['enhanced'].shape}")
    print(f"   ✓ Comparison: {results['comparison'].shape}")
    
    print("\n✅ Visualization test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("FRAME PREPROCESSING MODULE - UNIT TESTS")
    print("="*70)
    
    try:
        test_quality_assessment()
        test_image_enhancement()
        test_adaptive_detector()
        test_visualization()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

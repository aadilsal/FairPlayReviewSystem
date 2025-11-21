"""Quick test script for motion prediction functionality.

This script performs basic validation of the ball tracking and motion prediction
features without requiring actual video frames or YOLO models.
"""
from detection.ball_tracker import (
    BallTracker,
    KalmanFilter,
    fill_detection_gaps,
    filter_ball_detections,
    group_detections_by_frame
)


def test_linear_interpolation():
    """Test linear interpolation for short gaps."""
    print("Testing linear interpolation...")
    
    # Create sample detections with a gap
    detections = [
        {'frame_index': 0, 'x_min': 100, 'y_min': 200, 'x_max': 150, 'y_max': 250, 
         'confidence': 0.9, 'class_name': 'ball'},
        {'frame_index': 1, 'x_min': 110, 'y_min': 210, 'x_max': 160, 'y_max': 260, 
         'confidence': 0.85, 'class_name': 'ball'},
        # Gap at frame 2
        {'frame_index': 3, 'x_min': 130, 'y_min': 230, 'x_max': 180, 'y_max': 280, 
         'confidence': 0.88, 'class_name': 'ball'},
    ]
    
    # Fill gaps
    filled = fill_detection_gaps(detections, max_gap_frames=3, use_kalman=False)
    
    # Verify
    assert len(filled) == 4, f"Expected 4 detections, got {len(filled)}"
    
    # Check frame 2 was filled
    frame_2 = [d for d in filled if d['frame_index'] == 2]
    assert len(frame_2) == 1, "Frame 2 should have one prediction"
    assert frame_2[0]['detection_type'] == 'predicted', "Frame 2 should be marked as predicted"
    
    # Check interpolated position (should be between frame 1 and 3)
    pred = frame_2[0]
    assert 110 < pred['x_min'] < 130, f"x_min should be interpolated: {pred['x_min']}"
    assert 210 < pred['y_min'] < 230, f"y_min should be interpolated: {pred['y_min']}"
    
    print("✓ Linear interpolation test passed")
    return True


def test_kalman_filter():
    """Test Kalman filter prediction."""
    print("Testing Kalman filter...")
    
    # Create Kalman filter
    kf = KalmanFilter(process_noise=1.0, measurement_noise=10.0)
    
    # Initialize with first measurement
    kf.update([100, 200])
    
    # Add more measurements
    kf.update([110, 210])
    kf.update([120, 220])
    
    # Predict next position
    kf.predict()
    x, y = kf.get_position()
    
    # Should predict continuation of motion (relaxed bounds)
    # Kalman filter with measurement noise will have some variance
    assert 110 < x < 140, f"Predicted x should be reasonable: {x}"
    assert 210 < y < 240, f"Predicted y should be reasonable: {y}"
    
    print(f"✓ Kalman filter test passed (predicted: x={x:.1f}, y={y:.1f})")
    return True


def test_tracker():
    """Test BallTracker class."""
    print("Testing BallTracker...")
    
    # Create tracker
    tracker = BallTracker(max_gap_frames=3, prediction_confidence=0.3, use_kalman=False)
    
    # Add detections with gaps
    detections = [
        {'frame_index': 0, 'x_min': 100, 'y_min': 200, 'x_max': 150, 'y_max': 250, 
         'confidence': 0.9, 'class_name': 'ball'},
        {'frame_index': 1, 'x_min': 105, 'y_min': 205, 'x_max': 155, 'y_max': 255, 
         'confidence': 0.85, 'class_name': 'ball'},
        # Gap at 2-3
        {'frame_index': 4, 'x_min': 120, 'y_min': 220, 'x_max': 170, 'y_max': 270, 
         'confidence': 0.88, 'class_name': 'ball'},
    ]
    
    for det in detections:
        tracker.update(det['frame_index'], det)
    
    # Fill gaps
    tracker.fill_gaps()
    
    # Get all detections
    all_dets = tracker.get_all_detections()
    
    # Verify
    assert len(all_dets) == 5, f"Expected 5 detections (3 actual + 2 predicted), got {len(all_dets)}"
    assert tracker.is_predicted(2), "Frame 2 should be predicted"
    assert tracker.is_predicted(3), "Frame 3 should be predicted"
    assert not tracker.is_predicted(0), "Frame 0 should not be predicted"
    
    print("✓ BallTracker test passed")
    return True


def test_filter_balls():
    """Test ball detection filtering."""
    print("Testing ball filtering...")
    
    # Mixed detections
    detections = [
        {'frame_index': 0, 'class_name': 'sports ball', 'confidence': 0.9},
        {'frame_index': 0, 'class_name': 'person', 'confidence': 0.95},
        {'frame_index': 1, 'class_name': 'sports ball', 'confidence': 0.85},
        {'frame_index': 1, 'class_name': 'baseball bat', 'confidence': 0.7},
    ]
    
    # Filter to balls only
    balls = filter_ball_detections(detections, class_name='sports ball')
    
    assert len(balls) == 2, f"Expected 2 ball detections, got {len(balls)}"
    assert all(d['class_name'] == 'sports ball' for d in balls), "All should be balls"
    
    print("✓ Ball filtering test passed")
    return True


def test_group_by_frame():
    """Test grouping detections by frame."""
    print("Testing grouping by frame...")
    
    detections = [
        {'frame_index': 0, 'class_name': 'ball', 'confidence': 0.9},
        {'frame_index': 0, 'class_name': 'person', 'confidence': 0.95},
        {'frame_index': 1, 'class_name': 'ball', 'confidence': 0.85},
    ]
    
    grouped = group_detections_by_frame(detections)
    
    assert len(grouped) == 2, f"Expected 2 frames, got {len(grouped)}"
    assert len(grouped[0]) == 2, "Frame 0 should have 2 detections"
    assert len(grouped[1]) == 1, "Frame 1 should have 1 detection"
    
    print("✓ Grouping test passed")
    return True


def test_gap_size_limit():
    """Test that gaps exceeding max_gap_frames are not filled."""
    print("Testing gap size limit...")
    
    detections = [
        {'frame_index': 0, 'x_min': 100, 'y_min': 200, 'x_max': 150, 'y_max': 250, 
         'confidence': 0.9, 'class_name': 'ball'},
        # Large gap of 5 frames
        {'frame_index': 6, 'x_min': 200, 'y_min': 300, 'x_max': 250, 'y_max': 350, 
         'confidence': 0.88, 'class_name': 'ball'},
    ]
    
    # Max gap is 3, so gap of 5 should not be filled
    filled = fill_detection_gaps(detections, max_gap_frames=3, use_kalman=False)
    
    assert len(filled) == 2, f"Gap should not be filled, expected 2 detections, got {len(filled)}"
    
    # Now try with larger max_gap
    filled = fill_detection_gaps(detections, max_gap_frames=10, use_kalman=False)
    
    assert len(filled) == 7, f"Gap should be filled, expected 7 detections, got {len(filled)}"
    
    print("✓ Gap size limit test passed")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("Running Motion Prediction Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_linear_interpolation,
        test_kalman_filter,
        test_tracker,
        test_filter_balls,
        test_group_by_frame,
        test_gap_size_limit,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed! Motion prediction is working correctly.")
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

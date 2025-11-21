"""Configuration parameters for motion prediction.

This module provides sensible defaults and configuration options for
ball tracking with motion prediction.
"""

# Default configuration for motion prediction
MOTION_PREDICTION_CONFIG = {
    # Enable/disable motion prediction
    'enable_motion_prediction': True,
    
    # Maximum gap size (in frames) to fill with predictions
    # Recommended: 3-5 for typical videos, 10 for high-quality tracking
    'max_gap_frames': 5,
    
    # Confidence score assigned to predicted detections (0-1)
    # Should be lower than YOLO detection threshold to distinguish predictions
    'prediction_confidence': 0.3,
    
    # Use Kalman filtering for gaps > 3 frames
    # False: Use linear interpolation (fast, simple)
    # True: Use Kalman filter (smoother, more accurate)
    'use_kalman': False,
    
    # Kalman filter tuning (only used if use_kalman=True)
    'kalman_process_noise': 1.0,      # Motion model uncertainty
    'kalman_measurement_noise': 10.0,  # Detection uncertainty
}

# Preset configurations for different use cases
PRESETS = {
    # Conservative: Only fill very short gaps with high confidence
    'conservative': {
        'enable_motion_prediction': True,
        'max_gap_frames': 2,
        'prediction_confidence': 0.5,
        'use_kalman': False,
    },
    
    # Balanced: Good for most videos (default)
    'balanced': {
        'enable_motion_prediction': True,
        'max_gap_frames': 5,
        'prediction_confidence': 0.3,
        'use_kalman': False,
    },
    
    # Aggressive: Fill longer gaps, use Kalman filtering
    'aggressive': {
        'enable_motion_prediction': True,
        'max_gap_frames': 10,
        'prediction_confidence': 0.25,
        'use_kalman': True,
    },
    
    # High quality: Best predictions for smooth trajectories
    'high_quality': {
        'enable_motion_prediction': True,
        'max_gap_frames': 8,
        'prediction_confidence': 0.35,
        'use_kalman': True,
    },
    
    # Disabled: No motion prediction (original behavior)
    'disabled': {
        'enable_motion_prediction': False,
        'max_gap_frames': 0,
        'prediction_confidence': 0.3,
        'use_kalman': False,
    },
}


def get_config(preset='balanced'):
    """Get motion prediction configuration by preset name.
    
    Args:
        preset: One of 'conservative', 'balanced', 'aggressive', 'high_quality', 'disabled'
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = get_config('aggressive')
        >>> detect_balls(..., **config)
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")
    return PRESETS[preset].copy()


# Visualization colors (BGR format for OpenCV)
COLORS = {
    'detected': (0, 255, 0),      # Green for actual detections
    'predicted': (0, 255, 255),   # Yellow for predictions
    'high_conf': (0, 200, 0),     # Dark green for high confidence
    'low_conf': (0, 150, 255),    # Orange for low confidence
}


# Tuning guidelines
TUNING_GUIDE = """
Motion Prediction Tuning Guide
===============================

max_gap_frames:
  - 1-2:  Very conservative, only fill single-frame gaps
  - 3-5:  Balanced (recommended for most videos)
  - 6-10: Aggressive, good for high-quality footage
  - >10:  May produce unreliable predictions

prediction_confidence:
  - 0.1-0.2: Very uncertain predictions
  - 0.3-0.4: Balanced (default: 0.3)
  - 0.5+:    High confidence (use with conservative gap filling)

use_kalman:
  - False: Linear interpolation (fast, works well for short gaps)
  - True:  Kalman filter (smoother, better for longer gaps)

When to adjust:

1. Too many predictions:
   → Reduce max_gap_frames
   → Increase YOLO confidence threshold instead

2. Predictions in wrong positions:
   → Enable use_kalman
   → Reduce max_gap_frames
   → Tune kalman_measurement_noise (higher = trust detections less)

3. Missing important detections:
   → Increase max_gap_frames
   → Lower YOLO detection threshold
   → Improve ball detector training

4. Jittery predictions:
   → Enable use_kalman
   → Increase kalman_process_noise (smoother predictions)
   → Use 'high_quality' preset

5. Processing too slow:
   → Disable use_kalman
   → Reduce max_gap_frames
   → Use 'balanced' or 'conservative' preset
"""

if __name__ == '__main__':
    print("Motion Prediction Configuration")
    print("=" * 50)
    print("\nAvailable presets:")
    for name, config in PRESETS.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print(TUNING_GUIDE)

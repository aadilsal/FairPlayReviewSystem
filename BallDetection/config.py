
# Detection model configuration for dual-model architecture
DETECTION_CONFIG = {
    'model1_path': 'weights/ball-yolov8s.pt',  # High Precision (Model A)
    'model2_path': 'weights/yolov8_cricket_ball2/weights/best.pt',  # High Recall (Model B)
    'conf_threshold': 0.5,
    'iou_threshold': 0.45,
    'min_area': 25,
    'max_area': 8000,
    'aspect_ratio_min': 0.5,
    'aspect_ratio_max': 2.5,
    'ball_color': 'red',
    'enable_color_filter': False,
    'color_threshold': 0.2,
    'enable_motion_tracking': True,
    'min_velocity': 1,
    'max_trajectory_deviation': 100,
    'use_hybrid_tracking': True,
    'optical_flow_quality_threshold': 0.7,
    'physics_prediction_max_frames': 5,
    'gravity_constant': 0.5,
    'velocity_window_size': 5,
}

# ROI (Region of Interest) configuration for predictive cropping
ROI_CONFIG = {
    'BASE_CROP_SIZE': 150,           # Base crop size in pixels
    'VELOCITY_FACTOR': 2.0,         # Multiplier to expand crop based on speed
    'MAX_CROP_SIZE': 320,           # Maximum crop size in pixels
}

# State machine settings for detection pipeline
STATE_CONFIG = {
    'VALIDATION_FRAMES': 2,         # Frames required to confirm detection
    'MAX_MISS_STREAK': 5,           # Max consecutive missed frames before reset
}

POSTPROCESS_CONFIG = {
    'max_gap_to_fill': 10,
    'min_context_frames': 3,
    'velocity_window': 3,
    'enable_smoothing': True,
    'smoothing_window': 5,
    'smoothing_poly_order': 2,
    'validate_interpolation': True,
    'force_method': None,
    'log_corrections': True
}

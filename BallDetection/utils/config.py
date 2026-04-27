DETECTION_CONFIG = {
    'model2_path': 'weights/yolov8_cricket_ball2/weights/best.pt',  # High Precision (Model A)
    # PRIOTITY for model 2:
    # 1.weights/yolov8_cricket_ball2/weights/best.pt
    # 2.weights/ball-yolov8s.pt
    'model1_path': 'weights/ball_weights_new.pt',  # High Recall (Model B)
    'conf_threshold': 0.05,
    'iou_threshold': 0.1
}

FILTERS_CONFIG = {
    'enable_color_filter': True,
    'ball_color': 'white',
    'color_threshold': 0.2,
    'enable_area_filter': True,
    'min_area': 100,
    'max_area': 8000,
    'enable_aspect_ratio_filter': True,
    'aspect_ratio_min': 0.5,
    'aspect_ratio_max': 2.5,
    'enable_circularity_filter': True,
    'enable_shoe_filter': True,
}

ROI_CONFIG = {
    'BASE_CROP_SIZE': 200,
    'VELOCITY_FACTOR': 2.0,
    'MAX_CROP_SIZE': 800,
    'CROP_HEIGHT_MULTIPLIER': 2,
    'MAX_CROP_HEIGHT': 800,
}

STATE_CONFIG = {
    # Require a longer consistent run before locking into TRACKING.
    'VALIDATION_FRAMES': 4,
    # Minimum confidence gates for each state.
    'SCANNING_MIN_CONF': 0.15,
    'VALIDATION_MIN_CONF': 0.15,
    'TRACKING_MIN_CONF': 0.08,
    # Reject detections that jump too far between consecutive accepted points.
    'MAX_VALIDATION_JUMP_PX': 90.0,
    # Reject detections far from Kalman prediction while tracking.
    'MAX_TRACKING_JUMP_PX': 120.0,
    'MAX_MISS_STREAK': 5,
}


CROP_CONFIG = {
    'enable_dynamic_crop': True,
    'base_crop_size': 200,
    'velocity_factor': 2.0,
    'max_crop_size': 800,
    'crop_height_multiplier': 2,
    'max_crop_height': 800,
}

POST_PROCESSOR_CONFIG = {
    'enable_trajectory_overlay': True,
    'connect_detected_centers_only': True,
}


# ball_detector.py
    - Entry point for ball detection pipeline with state machine
    - BallDetector class: manages three detection states (SCANNING, VALIDATING, TRACKING)
    - STATE_SCANNING: full-frame YOLO detection, searches for ball candidates
    - STATE_VALIDATING: confirms detection persists for required frames via validation_counter
    - STATE_TRACKING: predicts next position using Kalman filter, crops ROI based on velocity
    - Uses BallKalmanInterpolator for position smoothing and prediction
    - detect(): main detection function - accepts frame and frame_idx
    - Uses global YOLOBallDetector via get_global_yolo_detector()
    - Calls yolo_detect_ball() or yolo_detect_ball_roi() depending on state
    - Filters detections via filter_and_select_ball_detection()
    - Uses ball_color from DETECTION_CONFIG
    - Returns ball_info dict: box, conf, interpolated_position
    - Handles missed detections with miss_streak counter, resets on MAX_MISS_STREAK
    - Maintains detection history and last_ball_info for continuity
    - Logs state transitions and detection results with frame index
    - reset(): clears state, validation counters, and Kalman filter
    - get_global_ball_detector(): singleton pattern for global detector instance

# config.py
    - Central configuration for dual-model ball detection pipeline
    - DETECTION_CONFIG: dual model paths (high precision + high recall), conf/iou thresholds
    - Area constraints: min_area (25px), max_area (8000px)
    - Aspect ratio filtering: min (0.5), max (2.5)
    - Color filtering: ball_color, enable_color_filter toggle, color_threshold (0.2)
    - Motion tracking: enable_motion_tracking, min_velocity, max_trajectory_deviation
    - Hybrid tracking: use_hybrid_tracking, optical_flow_quality_threshold (0.7)
    - Physics prediction: physics_prediction_max_frames (5), gravity_constant (0.5), velocity_window_size (5)
    - ROI_CONFIG: adaptive crop sizing (BASE_CROP_SIZE: 150px, MAX_CROP_SIZE: 320px), velocity-based expansion
    - STATE_CONFIG: validation frames (2), max miss streak (5) for state machine
    - POSTPROCESS_CONFIG: gap filling (10), smoothing (window: 5, poly_order: 2), interpolation validation
    - Used across all pipeline modules for consistent detection behavior

# filters.py
    - Validates YOLO detections to reduce false positives
    - is_shoe_like(): checks if bbox is shoe-shaped (high aspect ratio, near bottom of frame)
    - is_ball_circular(): analyzes contour circularity via Otsu thresholding and perimeter-area ratio
    - is_ball_colored(): HSV color filtering for white/red balls, configurable threshold (default 0.2)
    - filter_ball_detection(): applies shoe, circularity, and color filters in sequence
    - filter_and_select_ball_detection(): filters detections by area and aspect ratio constraints, returns highest confidence match

# interpolation.py
    - BallKalmanInterpolator: position smoothing via Kalman filter
    - State vector: [x, y, vx, vy] (position and velocity)
    - __init__(): configures filter matrices (F, H, P, R, Q) with standard Kalman parameters
    - update(): accepts measurement (x, y), performs prediction and update steps, returns smoothed position
    - predict_next(): predicts next position without measurement update, useful for tracking gaps
    - get_velocity(): returns current velocity vector (dx, dy) for motion prediction
    - reset(): clears filter state, optionally sets initial position for reinitialization
    - interpolate_trajectory(): legacy wrapper function for ball_infos list, processes detection history
    - Handles missing detections gracefully with Kalman predictions
    - Returns interpolated positions as tuples
    - Improves trajectory continuity and handles frame drops
    - Called from ball_detector to post-process detections and fill tracking gaps


# yolo_detect.py
    - YOLOBallDetector: YOLOv8 wrapper class for dual-model ball detection
    - __init__(): initializes two models (model1, model2) and device (CUDA/CPU), falls back to config paths if not provided
    - Model 1 (model1_path): high-precision ball detection for full-frame SCANNING state
    - Model 2 (model2_path): high-recall detection for ROI validation in VALIDATING/TRACKING states
    - Gracefully handles model loading failures, logs errors while continuing if one model unavailable
    - detect(): runs Model 1 inference on full frame, filters detections by ball-related class keywords
    - Converts YOLO tensor outputs to numpy arrays, extracts bbox (x, y, w, h), confidence, class_id
    - Handles variable class name formats (dict or list from model.names attribute)
    - detect_roi(): runs Model 2 inference on cropped ROI, maps detections back to global frame coordinates
    - Uses offset_coords to adjust detected positions relative to crop region
    - Both methods filter by confidence and IoU thresholds from config
    - Returns detections as tuples: (x, y, w, h, confidence, class_id)
    - get_global_yolo_detector(): singleton pattern, reinitializes if model paths change in config
    - yolo_detect_ball(): wrapper for full-frame detection using DETECTION_CONFIG thresholds and GLOBAL_CONFIG imgsz
    - yolo_detect_ball_roi(): wrapper for ROI detection with offset coordinate mapping
    - Called from ball_detector state machine for primary and validation detection pipelines


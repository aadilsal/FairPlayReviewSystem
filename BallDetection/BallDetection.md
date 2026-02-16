# ball_detector.py
    - Entry point for ball detection
    - Singleton YOLOBallDetector
    - get_yolo_detector(): model loading
    - detect_ball_on_frame(): main detection
    - Accepts frame, yolo_weights, debug, enable_preprocessing, ball_color
    - Uses DETECTION_CONFIG values
    - Calls YOLOBallDetector.detect()
    - Filters by area, aspect ratio
    - Applies is_shoe_like, is_ball_circular, is_ball_colored
    - Sorts detections by confidence
    - Returns ball_info dict: box, conf, source
    - Adds interpolated_position via BallKalmanInterpolator
    - Returns (frame, ball_info)
    - Logs detection results

# config.py
    - Central DETECTION_CONFIG, POSTPROCESS_CONFIG
    - Detection: conf_threshold, iou_threshold, imgsz
    - Area: min_area, max_area
    - Aspect ratio: aspect_ratio_min, aspect_ratio_max
    - Color: ball_color, enable_color_filter, color_threshold
    - Preprocessing: enable_preprocessing
    - Motion: enable_motion_tracking, min_velocity, max_trajectory_deviation
    - Hybrid: use_hybrid_tracking, optical_flow_quality_threshold
    - Physics: physics_prediction_max_frames, gravity_constant, velocity_window_size
    - Postprocess: max_gap_to_fill, min_context_frames, velocity_window
    - Smoothing: enable_smoothing, smoothing_window, smoothing_poly_order
    - Validation: validate_interpolation, log_corrections, force_method
    - Used by all pipeline files

# filters.py
    - Validates YOLO detections
    - is_shoe_like(): shape, bottom margin, elongation
    - is_ball_circular(): contour, circularity
    - is_ball_colored(): HSV mask, color ratio
    - All filters reduce false positives
    - Used in ball_detector pipeline

# interpolation.py
    - BallKalmanInterpolator: position smoothing
    - Kalman filter: [x, y, vx, vy]
    - update(): filter state, prediction
    - interpolate(): fills gaps, smooths trajectory
    - Handles missing positions
    - Returns interpolated positions
    - Improves trajectory continuity

# preprocessing.py
    - Frame enhancement for YOLO
    - estimate_blur(): Laplacian variance
    - preprocess_frame(): deblur, CLAHE, brightness, sharpen
    - Returns processed_frame, debug_info
    - preprocess_for_color_fallback(): aggressive contrast
    - Optional, controlled by config
    - Improves detection in blurry/small ball cases

# yolo_detect.py
    - YOLOBallDetector: YOLOv8 wrapper
    - __init__(): model, device
    - load_weights(): model loading, warm-up
    - detect(): inference, ball class filtering
    - Returns detections: (x, y, w, h, confidence, class_id)
    - Handles class name formats, tensor conversion
    - Called from ball_detector

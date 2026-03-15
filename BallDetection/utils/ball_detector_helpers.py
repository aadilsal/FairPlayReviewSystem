import logging
import numpy as np
from BallDetection.utils.config import STATE_CONFIG, ROI_CONFIG
from BallDetection.engines.yolo_detect import yolo_detect_ball, yolo_detect_ball_roi
from BallDetection.core.validator import filter_and_select_ball_detection

logger = logging.getLogger(__name__)

def handle_scanning_state(detector_instance, frame):
    """Initial search across the full frame."""
    yolo_detections = yolo_detect_ball(detector_instance.detector, frame)
    current_ball_info = filter_and_select_ball_detection(frame, yolo_detections)
    
    if current_ball_info:
        detector_instance.validation_counter = 1
        detector_instance.last_box = current_ball_info['box']
        # Initialize Kalman at detected spot
        detector_instance.kalman.reset(np.array(detector_instance.last_box[:2]))
        detector_instance.state = detector_instance.STATE_VALIDATING
        logger.info(f"[SCANNING] Candidate at {detector_instance.last_box[:2]}")
    
    return current_ball_info

def handle_validating_state(detector_instance, frame):
    """Confirming a candidate over multiple frames."""
    yolo_detections = yolo_detect_ball(detector_instance.detector, frame)
    current_ball_info = filter_and_select_ball_detection(frame, yolo_detections)

    if current_ball_info:
        detector_instance.kalman.predict_next()
        detector_instance.validation_counter += 1
        detector_instance.last_box = current_ball_info['box']
        detector_instance.kalman.update(np.array(detector_instance.last_box[:2]))

        if detector_instance.validation_counter >= STATE_CONFIG['VALIDATION_FRAMES']:
            detector_instance.state = detector_instance.STATE_TRACKING
            detector_instance.miss_streak = 0
            logger.info("[VALIDATING] Confirmed. Switching to TRACKING.")
    else:
        logger.info("[VALIDATING] Lost candidate. Resetting.")
        detector_instance.reset()
    
    return current_ball_info

def handle_tracking_state(detector_instance, frame):
    """Predictive ROI-based tracking."""
    # 1. Predict
    pred_x, pred_y = detector_instance.kalman.predict_next()
    velocity = detector_instance.kalman.get_velocity()
    speed = np.linalg.norm(velocity)

    # 2. Dynamic ROI Calculation
    crop_size = int(ROI_CONFIG['BASE_CROP_SIZE'] + ROI_CONFIG['VELOCITY_FACTOR'] * speed)
    crop_size = min(crop_size, ROI_CONFIG['MAX_CROP_SIZE'])
    crop_height = int(crop_size * ROI_CONFIG.get('CROP_HEIGHT_MULTIPLIER', 3))
    crop_height = min(crop_height, ROI_CONFIG.get('MAX_CROP_HEIGHT', 800))

    h, w = frame.shape[:2]
    x1 = max(0, int(pred_x) - crop_size // 2)
    x2 = min(w, int(pred_x) + crop_size // 2)
    y1 = max(0, int(pred_y) - crop_height // 2)
    y2 = min(h, int(pred_y) + crop_height // 2)
    roi_debug_box = [x1, y1, x2, y2]

    # 3. Detect in ROI
    current_ball_info = None
    if x2 > x1 and y2 > y1:
        frame_crop = frame[y1:y2, x1:x2]
        yolo_detections = yolo_detect_ball_roi(detector_instance.detector, frame_crop, (x1, y1))
        current_ball_info = filter_and_select_ball_detection(frame, yolo_detections)

    # 4. Update or Ghost
    if current_ball_info:
        detector_instance.kalman.update(np.array(current_ball_info['box'][:2]))
        detector_instance.last_box = current_ball_info['box']
        detector_instance.miss_streak = 0
    else:
        detector_instance.miss_streak += 1
        logger.warning(f"[TRACKING] Missed ({detector_instance.miss_streak})")
        
        # Create Ghost
        current_ball_info = {
            'box': [int(pred_x), int(pred_y), 0, 0],
            'conf': 0.0,
            'source': 'kalman-ghost',
            'ghost': True
        }

        if detector_instance.miss_streak >= STATE_CONFIG['MAX_MISS_STREAK']:
            detector_instance.reset()

    return current_ball_info, roi_debug_box

def finalize_detection_result(detector_instance, current_ball_info, roi_debug_box, frame_idx=0):
    """Processes final metadata and history logging."""
    detector_instance.last_ball_info = current_ball_info
    
    kf_pos = detector_instance.kalman.kf.x[:2]
    detector_instance.last_ball_info['interpolated_position'] = (float(kf_pos[0]), float(kf_pos[1]))
    
    if roi_debug_box:
        detector_instance.last_ball_info['roi_box'] = roi_debug_box

    # Enrich with metadata for post-processing tracking
    detector_instance.last_ball_info['frame_idx'] = frame_idx
    detector_instance.last_ball_info['state'] = detector_instance.state
    detector_instance.last_ball_info['miss_streak'] = detector_instance.miss_streak
    
    detector_instance.history.append(detector_instance.last_ball_info)
    return detector_instance.last_ball_info

def remap_to_original(result, x_offset):
    """
    Shift x-coordinates from cropped-frame space back to original-frame space.
    Called after finalize_detection_result so visualization overlays align
    with the full uncropped frame.
    """
    if x_offset == 0 or result is None:
        return result

    # Remap box x-coordinate (box is [x, y, w, h])
    if 'box' in result:
        result['box'][0] += x_offset

    # Remap ROI debug box (roi_box is [x1, y1, x2, y2])
    if 'roi_box' in result:
        result['roi_box'][0] += x_offset  # x1
        result['roi_box'][2] += x_offset  # x2

    # Remap interpolated position (tuple of (x, y))
    if 'interpolated_position' in result:
        ix, iy = result['interpolated_position']
        result['interpolated_position'] = (ix + x_offset, iy)

    return result

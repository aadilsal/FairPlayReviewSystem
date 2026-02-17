import logging
import numpy as np
from BallDetection.yolo_detect import yolo_detect_ball, yolo_detect_ball_roi, get_global_yolo_detector
from BallDetection.filters import filter_and_select_ball_detection
from BallDetection.interpolation import BallKalmanInterpolator
from BallDetection.config import DETECTION_CONFIG, ROI_CONFIG, STATE_CONFIG

logger = logging.getLogger(__name__)

class BallDetector:
    STATE_SCANNING = 0
    STATE_VALIDATING = 1
    STATE_TRACKING = 2

    def __init__(self):
        self.state = self.STATE_SCANNING
        self.detector = get_global_yolo_detector()
        self.kalman = BallKalmanInterpolator()
        self.validation_counter = 0
        self.miss_streak = 0
        self.last_ball_info = None
        self.last_box = None
        self.history = []

    def reset(self):
        self.state = self.STATE_SCANNING
        self.validation_counter = 0
        self.miss_streak = 0
        self.last_ball_info = None
        self.last_box = None
        self.kalman.reset()
        logger.info("BallDetector reset to SCANNING state.")

    def detect(self, frame, frame_idx=0):
        ball_color = DETECTION_CONFIG['ball_color']
        
        # Initialize result container
        current_ball_info = None
        roi_debug_box = None # To store [x1, y1, x2, y2]

        # ==========================
        # STATE: SCANNING
        # ==========================
        if self.state == self.STATE_SCANNING:
            yolo_detections = yolo_detect_ball(self.detector, frame)
            # Strict filtering for initial detection
            current_ball_info = filter_and_select_ball_detection(frame, yolo_detections, ball_color)
            
            if current_ball_info:
                self.validation_counter = 1
                self.last_box = current_ball_info['box']
                # Initialize Kalman at detected spot
                self.kalman.reset(np.array(self.last_box[:2]))
                self.state = self.STATE_VALIDATING
                logger.info(f"[SCANNING] Candidate at {self.last_box[:2]}")

        # ==========================
        # STATE: VALIDATING
        # ==========================
        elif self.state == self.STATE_VALIDATING:
            yolo_detections = yolo_detect_ball(self.detector, frame)
            current_ball_info = filter_and_select_ball_detection(frame, yolo_detections, ball_color)
            
            if current_ball_info:
                self.validation_counter += 1
                self.last_box = current_ball_info['box']
                # Update Kalman filter to refine velocity before tracking starts
                self.kalman.update(np.array(self.last_box[:2]))
                
                if self.validation_counter >= STATE_CONFIG['VALIDATION_FRAMES']:
                    self.state = self.STATE_TRACKING
                    self.miss_streak = 0
                    logger.info("[VALIDATING] Confirmed. Switching to TRACKING.")
            else:
                logger.info("[VALIDATING] Lost candidate. Resetting.")
                self.reset()

        # ==========================
        # STATE: TRACKING (Predictive ROI)
        # ==========================
        elif self.state == self.STATE_TRACKING:
            # 1. PREDICT: Advance Kalman State to t+1
            pred_x, pred_y = self.kalman.predict_next()
            velocity = self.kalman.get_velocity()
            speed = np.linalg.norm(velocity)

            # 2. CROP: Generate ROI based on Prediction + Velocity
            crop_size = int(ROI_CONFIG['BASE_CROP_SIZE'] + ROI_CONFIG['VELOCITY_FACTOR'] * speed)
            crop_size = min(crop_size, ROI_CONFIG['MAX_CROP_SIZE'])
            
            h, w = frame.shape[:2]
            x_c, y_c = int(pred_x), int(pred_y)
            
            x1 = max(0, x_c - crop_size // 2)
            y1 = max(0, y_c - crop_size // 2)
            x2 = min(w, x_c + crop_size // 2)
            y2 = min(h, y_c + crop_size // 2)
            
            # Store ROI coords for visualization
            roi_debug_box = [x1, y1, x2, y2]
            
            # 3. DETECT: Run Model B on the ROI
            if x2 > x1 and y2 > y1:
                frame_crop = frame[y1:y2, x1:x2]
                offset_coords = (x1, y1)
                yolo_detections = yolo_detect_ball_roi(self.detector, frame_crop, offset_coords)
                
                # Use RELAXED mode if available in your filters, otherwise standard
                current_ball_info = filter_and_select_ball_detection(frame, yolo_detections, ball_color)

            # 4. BRANCHING
            if current_ball_info:
                # HIT: Correct the Kalman Prediction
                self.kalman.update(np.array(current_ball_info['box'][:2]))
                self.last_box = current_ball_info['box']
                self.miss_streak = 0
            else:
                # MISS: Do NOT update Kalman (trust the prediction)
                self.miss_streak += 1
                logger.warning(f"[TRACKING] Missed ({self.miss_streak})")
                
                # Create GHOST detection at predicted location
                ghost_box = [int(pred_x), int(pred_y), 0, 0] # 0,0 size or copy previous size
                current_ball_info = {
                    'box': ghost_box,
                    'conf': 0.0,
                    'source': 'kalman-ghost',
                    'ghost': True
                }
                
                if self.miss_streak >= STATE_CONFIG['MAX_MISS_STREAK']:
                    self.reset()

        # ==========================
        # FINALIZE & RETURN
        # ==========================
        if current_ball_info:
            self.last_ball_info = current_ball_info
            
            # Add Interpolated Position (Clean float values for JSON)
            kf_pos = self.kalman.kf.x[:2]
            self.last_ball_info['interpolated_position'] = (float(kf_pos[0]), float(kf_pos[1]))
            
            # CRITICAL ADDITION: Add the ROI box to the result
            if roi_debug_box:
                self.last_ball_info['roi_box'] = roi_debug_box
            
            self.history.append(self.last_ball_info)
            return self.last_ball_info

        return None

# Singleton Pattern
_global_ball_detector = None
def get_global_ball_detector():
    global _global_ball_detector
    if _global_ball_detector is None:
        _global_ball_detector = BallDetector()
    return _global_ball_detector

def detect_ball(frame, frame_idx=0):
    return get_global_ball_detector().detect(frame, frame_idx)
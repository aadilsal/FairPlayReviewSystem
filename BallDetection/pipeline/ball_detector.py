import logging
import numpy as np
from BallDetection.core.interpolation import BallKalmanInterpolator
from BallDetection.utils.config import DETECTION_CONFIG, CROP_CONFIG
from BallDetection.engines.yolo_detect import get_global_yolo_detector

from BallDetection.utils.ball_detector_helpers import ( 
    handle_scanning_state,
    handle_validating_state,
    handle_tracking_state,
    finalize_detection_result,
    remap_to_original
)

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

        # Crop state (computed once on first frame, persists across reset())
        self.crop_initialized = False
        self.is_horizontal = False
        self.crop_x1 = 0
        self.crop_x2 = 0

    def _init_crop(self, frame):
        """Determine crop parameters from the first frame's aspect ratio."""
        if self.crop_initialized:
            return

        h, w = frame.shape[:2]
        self.crop_initialized = True

        if not CROP_CONFIG.get('ENABLE_AUTO_CROP', True):
            self.is_horizontal = False
            self.crop_x1 = 0
            self.crop_x2 = w
            logger.info("[CROP] Auto-crop disabled.")
            return

        if w > h:
            # Horizontal video — apply center crop
            self.is_horizontal = True
            left_pct = CROP_CONFIG.get('HORIZONTAL_CROP_LEFT_PCT', 0.2)
            right_pct = CROP_CONFIG.get('HORIZONTAL_CROP_RIGHT_PCT', 0.2)
            self.crop_x1 = int(w * left_pct)
            self.crop_x2 = int(w * (1.0 - right_pct))
            logger.info(f"[CROP] Horizontal video detected ({w}x{h}). "
                        f"Cropping x=[{self.crop_x1}:{self.crop_x2}] "
                        f"(removed {left_pct*100:.0f}% left, {right_pct*100:.0f}% right).")
        else:
            # Vertical video — no crop
            self.is_horizontal = False
            self.crop_x1 = 0
            self.crop_x2 = w
            logger.info(f"[CROP] Vertical video detected ({w}x{h}), no crop applied.")

    def _apply_crop(self, frame):
        """Return the cropped frame (or original if vertical)."""
        if self.is_horizontal:
            return frame[:, self.crop_x1:self.crop_x2]
        return frame

    def reset(self):
        self.state = self.STATE_SCANNING
        self.validation_counter = 0
        self.miss_streak = 0
        self.last_ball_info = None
        self.last_box = None
        self.kalman.reset()
        logger.info("BallDetector reset to SCANNING state.")

    def detect(self, frame, frame_idx=0):
        """
        Main entry point: Acts as a state handler and branch manager.
        Applies horizontal crop before detection, remaps results back to original coords.
        """
        # Initialize crop on first frame
        self._init_crop(frame)

        # Crop the frame for detection
        cropped_frame = self._apply_crop(frame)

        current_ball_info = None
        roi_debug_box = None

        # --- State Branching ---
        if self.state == self.STATE_SCANNING:
            current_ball_info = handle_scanning_state(self, cropped_frame)

        elif self.state == self.STATE_VALIDATING:
            current_ball_info = handle_validating_state(self, cropped_frame)

        elif self.state == self.STATE_TRACKING:
            current_ball_info, roi_debug_box = handle_tracking_state(self, cropped_frame)

        if current_ball_info:
            result = finalize_detection_result(self, current_ball_info, roi_debug_box, frame_idx)
            
            # Add crop bbox for visualization (Original Frame Coordinates)
            # Format: [x1, y1, x2, y2]
            h, w = frame.shape[:2]
            if self.is_horizontal:
                result['crop_box'] = [self.crop_x1, 0, self.crop_x2, h]
            else:
                # Even if not cropped, providing the box (full frame) might be useful or can be omitted.
                # User asked for "how much was cropped", so seeing the full frame box implies 0 crop.
                pass 

            # Remap cropped-frame coordinates back to original frame coordinates
            return remap_to_original(result, self.crop_x1)

        return None

_global_ball_detector = None

def get_global_ball_detector():
    global _global_ball_detector
    if _global_ball_detector is None:
        _global_ball_detector = BallDetector()
    return _global_ball_detector

def detect_ball(frame, frame_idx=0):
    return get_global_ball_detector().detect(frame, frame_idx)
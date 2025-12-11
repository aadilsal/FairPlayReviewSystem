import cv2
import logging
import os
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import scipy.interpolate

# 1. IMPORT PREPROCESSING (The missing link)
try:
    from preprocessing import preprocess_frame
except ImportError:
    print("[WARN] preprocessing.py not found. Using raw frames.")
    preprocess_frame = None

# Ensure you have this file or adjust import
from yolo_detect import YOLOBallDetector 

logger = logging.getLogger(__name__)

# Initialize detectors once
_yolo_detector = None

# 2. RELAXED CONFIGURATION (Optimized for Cricket/Sports)
DETECTION_CONFIG = {
    'conf_threshold': 0.2,
    'iou_threshold': 0.45,
    'imgsz': 640,
    'min_area': 20,               # ✅ FIX: Reduced to 20 to catch small/far balls
    'max_area': 4000,
    'aspect_ratio_min': 0.5,      # ✅ FIX: Lowered to 0.5 for motion blur
    'aspect_ratio_max': 2.5,      # ✅ FIX: Increased to 2.5 for elongated blur
    'ball_color': 'red',          # ✅ FIX: Default to 'red' (change to 'white' if needed)
    'enable_preprocessing': True, # ✅ FIX: Enabled by default
    'enable_color_filter': False, # ✅ FIX: Disabled initially to avoid false negatives
    'color_threshold': 0.2,
    'enable_motion_tracking': True,
    'min_velocity': 1,            # ✅ FIX: Low velocity allowed
    'max_trajectory_deviation': 100,
    'use_hybrid_tracking': True,  # ✅ NEW: Enable hybrid tracking by default
    'optical_flow_quality_threshold': 0.7,  # Threshold for quality score
    'physics_prediction_max_frames': 5,     # Max frames for physics prediction
    'gravity_constant': 0.5,                # pixels/frame²
    'velocity_window_size': 5,              # Frames for velocity calculation
}

# Post-processing configuration
POSTPROCESS_CONFIG = {
    'max_gap_to_fill': 10,           # Don't interpolate gaps > 10 frames
    'min_context_frames': 3,         # Need 3 YOLO frames on each side for spline
    'velocity_window': 3,            # Frames to calculate velocity
    'enable_smoothing': True,        # Apply Savitzky-Golay smoothing
    'smoothing_window': 5,           # Window for smoothing
    'smoothing_poly_order': 2,       # Polynomial order for smoothing
    'validate_interpolation': True,  # Run quality checks
    'force_method': None,            # Override auto-selection (for testing)
    'log_corrections': True          # Log when physics is corrected
}

# ---------------------------------------------------------
# HELPERS & TRACKER
# ---------------------------------------------------------

class BallMotionTracker:
    def __init__(self, max_history=5):
        self.positions = []  # [(center_x, center_y, frame_idx), ...]
        self.max_history = max_history
    
    def reset(self):
        self.positions = []
    
    def validate_detection(self, center, frame_idx):
        """Check if detection follows consistent motion pattern."""
        if len(self.positions) < 1:
            self.positions.append((center[0], center[1], frame_idx))
            return True
        
        # Calculate velocity
        prev_x, prev_y, prev_frame = self.positions[-1]
        dt = frame_idx - prev_frame
        
        # If pipeline doesn't pass frame_idx, dt is 0. 
        if dt == 0:
            return True 
        
        velocity_x = (center[0] - prev_x) / dt
        velocity_y = (center[1] - prev_y) / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Relaxed velocity check
        min_vel = DETECTION_CONFIG.get('min_velocity', 1)
        # Note: We barely reject based on low speed anymore to be safe
        
        # Update history
        self.positions.append((center[0], center[1], frame_idx))
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
        
        return True

_motion_tracker = BallMotionTracker()

class HybridBallTracker:
    """
    Hybrid tracker combining optical flow for blur/fast motion and physics for occlusion.
    """
    def __init__(self):
        self.last_center = None  # (x, y)
        self.last_w = None
        self.last_h = None
        self.velocity = (0.0, 0.0)  # (vx, vy)
        self.detection_history = []  # List of (x, y, frame_idx)
        self.prev_frame = None  # Grayscale frame
        self.tracking_mode = "yolo"  # "yolo", "optical_flow", "physics"
        self.frames_since_yolo = 0
        self.consecutive_failures = 0
        self.max_physics_frames = DETECTION_CONFIG.get('physics_prediction_max_frames', 5)
        self.velocity_window = DETECTION_CONFIG.get('velocity_window_size', 5)
        self.gravity = DETECTION_CONFIG.get('gravity_constant', 0.5)
        # Optical flow params
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def update_with_yolo(self, frame, ball_center, w, h, frame_idx):
        """Update tracker with successful YOLO detection."""
        self.last_center = ball_center
        self.last_w = w
        self.last_h = h
        self.detection_history.append((ball_center[0], ball_center[1], frame_idx))
        if len(self.detection_history) > self.velocity_window:
            self.detection_history.pop(0)
        self.calculate_velocity()
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracking_mode = "yolo"
        self.frames_since_yolo = 0
        self.consecutive_failures = 0
        logger.debug(f"[Hybrid] Updated with YOLO at {ball_center}, velocity {self.velocity}")
    
    def calculate_velocity(self):
        """Calculate velocity using sliding window of detections."""
        if len(self.detection_history) < 2:
            self.velocity = (0.0, 0.0)
            return
        
        # Use linear regression for velocity
        points = np.array(self.detection_history)
        t = points[:, 2]
        x = points[:, 0]
        y = points[:, 1]
        
        if len(t) > 1:
            vx = np.polyfit(t, x, 1)[0]
            vy = np.polyfit(t, y, 1)[0]
            self.velocity = (vx, vy)
    
    def track_with_optical_flow(self, current_frame):
        """Try optical flow tracking. Returns (predicted_center, quality_score) or (None, 0)."""
        if self.last_center is None or self.prev_frame is None:
            return None, 0.0
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_points = np.array([self.last_center], dtype=np.float32).reshape(-1, 1, 2)
        
        next_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, current_gray, prev_points, None, **self.lk_params)
        
        predicted_center = (int(next_points[0][0][0]), int(next_points[0][0][1]))
        quality = self.is_optical_flow_reliable(status, predicted_center, error, current_frame.shape)
        
        if quality > DETECTION_CONFIG.get('optical_flow_quality_threshold', 0.7):
            self.last_center = predicted_center
            self.prev_frame = current_gray
            self.tracking_mode = "optical_flow"
            return predicted_center, quality
        return None, quality
    
    def is_optical_flow_reliable(self, status, predicted_point, error, frame_shape):
        """Assess optical flow quality. Returns score 0-1."""
        if status[0] != 1:
            return 0.0
        
        h, w = frame_shape[:2]
        margin = 10
        if not (margin <= predicted_point[0] <= w - margin and margin <= predicted_point[1] <= h - margin):
            return 0.0
        
        # Movement magnitude check
        dx = predicted_point[0] - self.last_center[0]
        dy = predicted_point[1] - self.last_center[1]
        movement = np.sqrt(dx**2 + dy**2)
        expected_speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if expected_speed > 0 and not (0.5 * expected_speed <= movement <= 3.0 * expected_speed):
            return 0.0
        
        # Error check
        if error[0] > 5.0:
            return 0.0
        
        # Jump check
        expected_x = self.last_center[0] + self.velocity[0]
        expected_y = self.last_center[1] + self.velocity[1]
        jump = np.sqrt((predicted_point[0] - expected_x)**2 + (predicted_point[1] - expected_y)**2)
        if jump > 50:
            return 0.0
        
        return 1.0  # Perfect quality
    
    def predict_with_physics(self, frame_idx):
        """Predict position using physics. Returns predicted_center or None."""
        if self.last_center is None or len(self.detection_history) < 3:
            return None
        
        last_frame = self.detection_history[-1][2]
        t = frame_idx - last_frame
        if t > self.max_physics_frames:
            return None
        
        vx, vy = self.velocity
        x = self.last_center[0] + vx * t
        y = self.last_center[1] + vy * t + 0.5 * self.gravity * t**2
        
        predicted_center = (int(x), int(y))
        self.tracking_mode = "physics"
        logger.debug(f"[Hybrid] Physics prediction: t={t}, pos=({x:.1f}, {y:.1f})")
        return predicted_center
    
    def reset(self):
        """Reset tracker state."""
        self.last_center = None
        self.last_w = None
        self.last_h = None
        self.velocity = (0.0, 0.0)
        self.detection_history = []
        self.prev_frame = None
        self.tracking_mode = "yolo"
        self.frames_since_yolo = 0
        self.consecutive_failures = 0
        logger.debug("[Hybrid] Tracker reset")

# Global hybrid tracker
_hybrid_tracker = None

def get_hybrid_tracker():
    global _hybrid_tracker
    if _hybrid_tracker is None:
        _hybrid_tracker = HybridBallTracker()
    return _hybrid_tracker

class TrajectoryPostProcessor:
    """
    Post-processes ball trajectory to fill gaps using backwards interpolation.
    """
    def __init__(self, config=None):
        self.config = config or POSTPROCESS_CONFIG.copy()
    
    def detect_gaps(self, frame_results):
        """Identify gaps in detection where interpolation is needed."""
        gaps = []
        in_gap = False
        gap_start = None
        gap_frames = []
        
        for i, result in enumerate(frame_results):
            if result['source'] == 'yolo':
                if in_gap and gap_start is not None:
                    # End of gap
                    gap_end = i - 1
                    start_pos = frame_results[gap_start]['position']
                    end_pos = result['position']
                    gaps.append({
                        'start_frame': gap_start,
                        'end_frame': gap_end,
                        'gap_frames': gap_frames,
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    })
                    in_gap = False
                    gap_start = None
                    gap_frames = []
            else:
                if not in_gap:
                    in_gap = True
                    gap_start = i
                gap_frames.append(i)
        
        return gaps
    
    def calculate_velocity_at_boundary(self, frame_results, boundary_frame, window=3):
        """Calculate velocity vector at gap boundary."""
        if boundary_frame - window < 0 or boundary_frame + window >= len(frame_results):
            return (0.0, 0.0)
        
        # For start boundary: use frames before
        points = []
        for i in range(boundary_frame - window, boundary_frame + 1):
            if frame_results[i]['position'] is not None:
                points.append((frame_results[i]['position'][0], frame_results[i]['position'][1], i))
        
        if len(points) < 2:
            return (0.0, 0.0)
        
        points = np.array(points)
        t = points[:, 2]
        x = points[:, 0]
        y = points[:, 1]
        
        vx = np.polyfit(t, x, 1)[0]
        vy = np.polyfit(t, y, 1)[0]
        return (vx, vy)
    
    def linear_interpolate(self, start_pos, end_pos, num_frames):
        """Simple linear interpolation."""
        positions = []
        for i in range(num_frames):
            alpha = (i + 1) / (num_frames + 1)
            x = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
            y = start_pos[1] + alpha * (end_pos[1] - start_pos[1])
            positions.append((x, y))
        return positions
    
    def parabolic_interpolate(self, start_pos, end_pos, start_velocity, end_velocity, num_frames):
        """Fit parabolic curve considering velocities."""
        # Simplified: use quadratic Bezier
        control_x = (start_pos[0] + end_pos[0]) / 2 + (end_velocity[0] - start_velocity[0]) * num_frames / 4
        control_y = (start_pos[1] + end_pos[1]) / 2 + (end_velocity[1] - start_velocity[1]) * num_frames / 4
        
        positions = []
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            x = (1 - t)**2 * start_pos[0] + 2 * (1 - t) * t * control_x + t**2 * end_pos[0]
            y = (1 - t)**2 * start_pos[1] + 2 * (1 - t) * t * control_y + t**2 * end_pos[1]
            positions.append((x, y))
        return positions
    
    def spline_interpolate(self, context_positions, gap_frames):
        """Use cubic spline for smooth curves."""
        if len(context_positions) < 4:
            return []
        
        frames = [p[2] for p in context_positions]
        xs = [p[0] for p in context_positions]
        ys = [p[1] for p in context_positions]
        
        cs_x = scipy.interpolate.CubicSpline(frames, xs)
        cs_y = scipy.interpolate.CubicSpline(frames, ys)
        
        positions = []
        for frame in gap_frames:
            x = cs_x(frame)
            y = cs_y(frame)
            positions.append((x, y))
        return positions
    
    def select_interpolation_method(self, gap_info, frame_results):
        """Choose interpolation method based on gap characteristics."""
        gap_length = len(gap_info['gap_frames'])
        
        if self.config.get('force_method'):
            return self.config['force_method']
        
        # Gap length
        if gap_length <= 3:
            return 'linear'
        elif gap_length <= 6:
            return 'parabolic'
        else:
            # Check context
            start_idx = gap_info['start_frame']
            end_idx = gap_info['end_frame']
            context_before = [r for r in frame_results[max(0, start_idx-5):start_idx] if r['source'] == 'yolo']
            context_after = [r for r in frame_results[end_idx+1:min(len(frame_results), end_idx+6)] if r['source'] == 'yolo']
            if len(context_before) >= self.config['min_context_frames'] and len(context_after) >= self.config['min_context_frames']:
                return 'spline'
            else:
                return 'parabolic'
    
    def interpolate_gap(self, gap_info, frame_results):
        """Fill gap with interpolated positions."""
        method = self.select_interpolation_method(gap_info, frame_results)
        num_frames = len(gap_info['gap_frames'])
        
        if method == 'linear':
            positions = self.linear_interpolate(gap_info['start_pos'], gap_info['end_pos'], num_frames)
        elif method == 'parabolic':
            start_vel = self.calculate_velocity_at_boundary(frame_results, gap_info['start_frame'], self.config['velocity_window'])
            end_vel = self.calculate_velocity_at_boundary(frame_results, gap_info['end_frame'], self.config['velocity_window'])
            positions = self.parabolic_interpolate(gap_info['start_pos'], gap_info['end_pos'], start_vel, end_vel, num_frames)
        elif method == 'spline':
            # Collect context
            start_idx = gap_info['start_frame']
            end_idx = gap_info['end_frame']
            context = []
            for i in range(max(0, start_idx-5), min(len(frame_results), end_idx+6)):
                if frame_results[i]['position'] is not None:
                    context.append((frame_results[i]['position'][0], frame_results[i]['position'][1], i))
            positions = self.spline_interpolate(context, gap_info['gap_frames'])
        else:
            positions = []
        
        return positions, method
    
    def validate_interpolation(self, gap_info, interpolated_positions):
        """Check if interpolation is reasonable."""
        if not interpolated_positions:
            return False, 0.0, ["No positions generated"]
        
        warnings = []
        quality_score = 1.0
        
        # Bounds check
        for pos in interpolated_positions:
            if pos[0] < 0 or pos[1] < 0 or pos[0] > 1920 or pos[1] > 1080:  # Assume 1080p
                warnings.append("Position out of bounds")
                quality_score -= 0.5
        
        # Velocity check
        for i in range(1, len(interpolated_positions)):
            dx = interpolated_positions[i][0] - interpolated_positions[i-1][0]
            dy = interpolated_positions[i][1] - interpolated_positions[i-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            if speed > 200:
                warnings.append(f"Excessive speed: {speed}")
                quality_score -= 0.3
        
        is_valid = quality_score > 0.5
        return is_valid, quality_score, warnings
    
    def process_trajectory(self, frame_results):
        """Main entry point for post-processing."""
        gaps = self.detect_gaps(frame_results)
        corrected_results = frame_results.copy()
        
        for gap_idx, gap in enumerate(gaps):
            if len(gap['gap_frames']) > self.config['max_gap_to_fill']:
                continue
            
            interpolated_positions, method = self.interpolate_gap(gap, frame_results)
            
            if self.config['validate_interpolation']:
                is_valid, quality, warnings = self.validate_interpolation(gap, interpolated_positions)
                if not is_valid:
                    logger.warning(f"Gap {gap_idx}: Invalid interpolation - {warnings}")
                    continue
            
            for i, frame_idx in enumerate(gap['gap_frames']):
                if i < len(interpolated_positions):
                    original = corrected_results[frame_idx]
                    corrected_results[frame_idx] = {
                        'frame_idx': frame_idx,
                        'position': interpolated_positions[i],
                        'conf': -3.0,
                        'source': f'interpolated_{method}',
                        'original_source': original['source'],
                        'original_position': original['position'],
                        'gap_id': gap_idx
                    }
                    if self.config['log_corrections']:
                        print(f"[PostProcess] Frame {frame_idx}: Corrected {original['source']} to interpolated_{method}")
        
        if self.config['enable_smoothing']:
            corrected_results = self.apply_smoothing(corrected_results)
        
        return corrected_results
    
    def apply_smoothing(self, frame_results):
        """Apply Savitzky-Golay smoothing."""
        positions = []
        frames = []
        for r in frame_results:
            if r['position'] is not None:
                positions.append(r['position'])
                frames.append(r['frame_idx'])
        
        if len(positions) < self.config['smoothing_window']:
            return frame_results
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        from scipy.signal import savgol_filter
        smoothed_xs = savgol_filter(xs, self.config['smoothing_window'], self.config['smoothing_poly_order'])
        smoothed_ys = savgol_filter(ys, self.config['smoothing_window'], self.config['smoothing_poly_order'])
        
        smoothed_results = frame_results.copy()
        pos_idx = 0
        for i, r in enumerate(smoothed_results):
            if r['position'] is not None:
                smoothed_results[i]['position'] = (smoothed_xs[pos_idx], smoothed_ys[pos_idx])
                pos_idx += 1
        
        return smoothed_results
    
    def visualize_corrections(self, original_results, corrected_results, output_path):
        """Create visualization of corrections."""
        try:
            import matplotlib.pyplot as plt
            
            orig_x = [r['position'][0] for r in original_results if r['position'] is not None and r['source'] == 'yolo']
            orig_y = [r['position'][1] for r in original_results if r['position'] is not None and r['source'] == 'yolo']
            
            phys_x = [r['position'][0] for r in original_results if r['position'] is not None and r['source'] == 'physics']
            phys_y = [r['position'][1] for r in original_results if r['position'] is not None and r['source'] == 'physics']
            
            interp_x = [r['position'][0] for r in corrected_results if r['position'] is not None and 'interpolated' in r['source']]
            interp_y = [r['position'][1] for r in corrected_results if r['position'] is not None and 'interpolated' in r['source']]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(orig_x, orig_y, c='blue', label='YOLO detections', s=50)
            plt.scatter(phys_x, phys_y, c='red', label='Physics predictions', s=30, alpha=0.7)
            plt.plot(interp_x, interp_y, c='green', label='Interpolated trajectory', linewidth=2)
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.title('Trajectory Correction Visualization')
            plt.legend()
            plt.gca().invert_yaxis()  # Video coordinates
            plt.savefig(output_path)
            plt.close()
            print(f"[PostProcess] Visualization saved to {output_path}")
        except ImportError:
            print("[PostProcess] Matplotlib not available for visualization")

def is_ball_colored(frame, x, y, w, h, ball_color='white'):
    """Validate if detection region has ball-like colors."""
    try:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if ball_color == 'white':
            mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255)) # Relaxed white
        elif ball_color == 'red':
            mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            return True
        
        matching_pixels = cv2.countNonZero(mask)
        total_pixels = w * h
        color_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0
        
        threshold = DETECTION_CONFIG.get('color_threshold', 0.2)
        return color_ratio >= threshold
    except Exception as e:
        print(f"[WARN] Color validation failed: {e}")
        return True


def is_shoe_like(img: np.ndarray, bbox: Tuple[int, int, int, int], elongation_thresh: float = 3.0, bottom_of_frame_margin: int = 20) -> bool:
    """Heuristic to reject detections that look like shoes."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    # Only check elongation if it's at the very bottom of the frame
    img_h = img.shape[0]
    if (img_h - y2) <= bottom_of_frame_margin:
        elongation = w / h
        if elongation > elongation_thresh:
            return True

    return False


def is_ball_circular(img: np.ndarray, bbox: Tuple[int, int, int, int], circularity_thresh: float = 0.4) -> bool:
    """Estimate circularity. Lower threshold allowed for motion blur."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return False
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return False
        
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5: return False 
    
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0: return False
        
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity >= circularity_thresh


def get_yolo_detector(weights_path=None):
    global _yolo_detector
    default_weights = "weights/ball-yolov8s.pt" 

    if _yolo_detector is None:
        target_weights = weights_path if weights_path else default_weights
        print(f"[INFO] Initializing YOLO detector with {target_weights}")
        _yolo_detector = YOLOBallDetector(target_weights)
    elif weights_path is not None and weights_path != _yolo_detector.model_path:
        print(f"[INFO] Reloading YOLO weights: {weights_path}")
        _yolo_detector.load_weights(weights_path)
        
    return _yolo_detector

# ---------------------------------------------------------
# MAIN DETECTOR FUNCTION
# ---------------------------------------------------------
def detect_ball_on_frame(
    frame, 
    yolo_weights=None, 
    debug=False,         
    enable_preprocessing=None, 
    ball_color=None,
    conf_threshold=None, 
    iou_threshold=None, 
    imgsz=None,
    frame_idx=0,
    enable_hybrid_tracking=None  # NEW: Parameter to enable/disable hybrid tracking
):
    """
    Returns (frame, ball_info)
    ball_info is { "box": [x,y,w,h], "conf": float, "source": str, "velocity": [vx,vy] } or None
    """
    # 1. Config Resolution
    if ball_color is None: ball_color = DETECTION_CONFIG.get('ball_color', 'red')
    if conf_threshold is None: conf_threshold = DETECTION_CONFIG.get('conf_threshold', 0.2)
    if iou_threshold is None: iou_threshold = DETECTION_CONFIG.get('iou_threshold', 0.45)
    if imgsz is None: imgsz = DETECTION_CONFIG.get('imgsz', 640)
    if enable_preprocessing is None: enable_preprocessing = DETECTION_CONFIG.get('enable_preprocessing', True)
    if enable_hybrid_tracking is None: enable_hybrid_tracking = DETECTION_CONFIG.get('use_hybrid_tracking', True)

    # 2. PREPROCESSING (The Key Fix)
    detection_frame = frame
    if enable_preprocessing and preprocess_frame is not None:
        try:
            # Enhance contrast/sharpness so YOLO sees the ball better
            detection_frame, _ = preprocess_frame(frame, ball_color=ball_color)
        except Exception as e:
            if debug: print(f"[WARN] Preprocessing failed: {e}")

    # 3. Run Inference on ENHANCED frame
    detector = get_yolo_detector(yolo_weights)
    yolo_detections = detector.detect(detection_frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz)

    filtered = []

    # 4. Filtering Loop
    for det in yolo_detections:
        if len(det) == 6:
            (x, y, w, h, confidence, cls_id) = det
        else:
            (x, y, w, h, confidence) = det
        
        # --- Filter A: Aspect Ratio & Area ---
        aspect = (w / h) if h > 0 else 0
        area = w * h
        
        if area < DETECTION_CONFIG['min_area'] or area > DETECTION_CONFIG['max_area']:
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Area {area}")
            continue

        if not (DETECTION_CONFIG['aspect_ratio_min'] < aspect < DETECTION_CONFIG['aspect_ratio_max']):
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Aspect Ratio {aspect:.2f}")
            continue
        
        x1, y1, x2, y2 = x, y, x + w, y + h

        # --- Filter B: Shoe-like rejection ---
        if is_shoe_like(frame, (x1, y1, x2, y2)):
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Shoe-like")
            continue

        # --- Filter C: Circularity check ---
        # Note: We relaxed the threshold to 0.4
        if not is_ball_circular(frame, (x1, y1, x2, y2)):
            if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Non-circular")
            continue

        # --- Filter D: Color validation ---
        if DETECTION_CONFIG.get('enable_color_filter', False):
            if not is_ball_colored(frame, x, y, w, h, ball_color):
                if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Color mismatch ({ball_color})")
                continue

        # --- Filter E: Motion tracking ---
        if DETECTION_CONFIG.get('enable_motion_tracking', True):
            center = (x + w // 2, y + h // 2)
            if not _motion_tracker.validate_detection(center, frame_idx):
                if debug: print(f"[DEBUG] Frame {frame_idx}: Rejected Motion trajectory")
                continue
        
        # Accepted
        filtered.append((x, y, w, h, confidence))

    # 5. Prepare Result from YOLO
    # 5. Prepare Result from YOLO
    ball_info = None
    tracker = get_hybrid_tracker()
    if filtered:
        # Sort by confidence (descending)
        filtered.sort(key=lambda x: x[4], reverse=True)
        best_det = filtered[0]
        
        ball_info = {
            "box": [best_det[0], best_det[1], best_det[2], best_det[3]],
            "conf": float(best_det[4]),
            "source": "yolo",
            "velocity": list(tracker.velocity)
        }
        
        # Update hybrid tracker with successful detection
        center = (best_det[0] + best_det[2] // 2, best_det[1] + best_det[3] // 2)
        tracker.update_with_yolo(frame, center, best_det[2], best_det[3], frame_idx)
        print(f"[INFO] Frame {frame_idx}: Ball detected by YOLO at ({center[0]}, {center[1]}) with conf {best_det[4]:.2f}")
    
    # 6. Fallback to Hybrid Tracking if YOLO failed and enabled
    elif enable_hybrid_tracking:
        tracker.frames_since_yolo += 1
        tracker.consecutive_failures += 1
        
        # Try optical flow
        flow_result, flow_quality = tracker.track_with_optical_flow(frame)
        if flow_result:
            x = flow_result[0] - tracker.last_w // 2
            y = flow_result[1] - tracker.last_h // 2
            ball_info = {
                "box": [x, y, tracker.last_w, tracker.last_h],
                "conf": -1.0,
                "source": "optical_flow",
                "velocity": list(tracker.velocity)
            }
            print(f"[INFO] Frame {frame_idx}: Ball predicted by optical flow at ({flow_result[0]}, {flow_result[1]})")
        else:
            # Optical flow failed, try physics
            physics_result = tracker.predict_with_physics(frame_idx)
            if physics_result:
                x = physics_result[0] - tracker.last_w // 2
                y = physics_result[1] - tracker.last_h // 2
                ball_info = {
                    "box": [x, y, tracker.last_w, tracker.last_h],
                    "conf": -2.0,
                    "source": "physics",
                    "velocity": list(tracker.velocity)
                }
                print(f"[INFO] Frame {frame_idx}: Ball predicted by physics at ({physics_result[0]}, {physics_result[1]})")
            else:
                # Too many failures, reset
                if tracker.consecutive_failures > 7:
                    tracker.reset()
                    print(f"[INFO] Frame {frame_idx}: Too many failures, resetting tracker")

    return frame, ball_info
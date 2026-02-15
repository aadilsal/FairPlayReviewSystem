"""Coordinate system (image-space, pixels):
- Origin at top-left.
- +x to the right, +y down.
- Gravity acts in +y.
- Delivery direction (bowler -> batsman) is +x by convention.
"""

import cv2
import logging
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

# Optional Imports
try:
    from preprocessing import preprocess_frame
except ImportError:
    preprocess_frame = None

try:
    from trajectory_fitting import fit_trajectory, detect_events_from_trajectory
except ImportError:
    fit_trajectory = None
    detect_events_from_trajectory = None

try:
    from yolo_detect import YOLOBallDetector
except ImportError:
    YOLOBallDetector = None

try:
    from kalman_filter import BallKalmanFilter
except ImportError:
    BallKalmanFilter = None

try:
    from pitch_plane import PitchPlaneEstimator
except ImportError:
    PitchPlaneEstimator = None

logger = logging.getLogger(__name__)

# CONFIGURATION
# Locked physics parameters: edit here only; no per-video tuning.
DETECTION_CONFIG = {
    'conf_threshold': 0.2,
    'iou_threshold': 0.45,
    'imgsz': 640,
    'min_area': 20,
    'max_area': 4000,
    'aspect_ratio_min': 0.5,
    'aspect_ratio_max': 2.5,
    'bootstrap_frames': 5,          # Require one more frame of consistency
    'bootstrap_conf_threshold': 0.4, # Stricter conf for initial lock
    'min_bootstrap_motion': 3.0,     # Must move at least 3 pixels between frames to lock
    'optical_flow_quality_threshold': 0.7,
    'max_drift_pixels': 20,
    'drift_check_interval': 12,
    'roi_margin': 50,
    'max_coast_frames': 25,
    'stuck_detection_threshold': 5.0,
    'stuck_frames': 8,
    'min_velocity_flight': 2.0,
    'max_velocity': 200.0,
    'kalman_diverrence_threshold': 80.0,
    'recovery_roi_size': 300,
    'yolo_recovery_conf': 0.15,
    'min_downward_velocity': 4.0,
    'min_upward_velocity': 1.5,
    'pitch_warmup_frames': 10,
    'pitch_variance_threshold': 12.0,
    'pitch_min_static_ratio': 0.45,
    'pitch_prefer_lower_half': 0.55,
    'pitch_texture_threshold': 18.0,
    'pitch_edge_low': 40,
    'pitch_edge_high': 140,
    'pitch_margin': 2.0,
    'bounce_damping': 0.75,
    'fit_window': 12,
    'max_fit_gap': 5,
    'fit_residual_threshold': 35.0,
    'yolo_gate_px': 70.0,
    'optical_gate_px': 45.0,
    'csrt_gate_px': 25.0,
    'tld_gate_px': 55.0,
    'yolo_noise_scale': 1.0,
    'optical_noise_scale': 2.5,
    'csrt_noise_scale': 6.0,
    'tld_noise_scale': 5.0,
    'guided_noise_scale': 4.0,
    'high_confidence': 0.65,
    'confidence_decay': 0.03,
    'min_confidence': 0.2,
    'max_physics_violations': 5,
    'physics_prediction_horizon': 10,
    'impact_no_measurement_frames': 3,
    'impact_roi_margin': 12,
    'post_impact_predict_frames': 120,
}

STATE_BOOTSTRAP = "BOOTSTRAP"
STATE_TRACKING = "TRACKING"
STATE_LOST = "LOST"

@dataclass
class BallState:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    has_bounced: bool
    confidence: float
    impact_frame: Optional[int] = None
    impact_point: Optional[Tuple[float, float]] = None
    would_hit_stumps: Optional[bool] = None
    hit_frame: Optional[int] = None
    hit_point: Optional[Tuple[float, float]] = None
    bounce_point: Optional[Tuple[float, float]] = None
    post_impact_path: Optional[List[Tuple[float, float]]] = None

    def to_dict(self):
        return {
            "position": [float(self.position[0]), float(self.position[1])],
            "velocity": [float(self.velocity[0]), float(self.velocity[1])],
            "acceleration": [float(self.acceleration[0]), float(self.acceleration[1])],
            "has_bounced": bool(self.has_bounced),
            "confidence": float(self.confidence),
            "impact_frame": self.impact_frame,
            "impact_point": ([float(self.impact_point[0]), float(self.impact_point[1])] if self.impact_point else None),
            "would_hit_stumps": self.would_hit_stumps,
            "hit_frame": self.hit_frame,
            "hit_point": ([float(self.hit_point[0]), float(self.hit_point[1])] if self.hit_point else None),
            "bounce_point": ([float(self.bounce_point[0]), float(self.bounce_point[1])] if self.bounce_point else None),
            "post_impact_path": (
                [[float(p[0]), float(p[1])] for p in self.post_impact_path]
                if self.post_impact_path else None
            )
        }

class HybridBallTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = STATE_BOOTSTRAP
        self.kf = None
        self.kf_initialized = False
        self.csrt = None
        self.tld = None
        self.last_center = None
        self.last_box = None
        self.prev_frame_gray = None
        self.prev_physics_pos = None
        self.physics_violations = 0
        self.consecutive_detections = 0
        self.frames_without_update = 0
        self.frames_without_measurement = 0
        self.position_history = []
        self.full_track_history = []
        self.velocity_history = []
        self.uncertainties = []
        self.pre_bounce_history = []
        self.post_bounce_history = []
        self.bounce_detected = False
        self.bounce_frame_idx = -1
        self.stuck_counter = 0
        self.confidence = 0.0
        self.last_measurement_frame = -1
        self.pitch_estimator = None
        self.pitch_ready = False
        self.pitch_y = None
        self.pitch_model = None
        self.bootstrap_history = []
        self.batsman_box = None
        self.wicket_line = None
        self.impact_frame = None
        self.impact_point = None
        self.hit_frame = None
        self.hit_point = None
        self.would_hit_stumps = None
        self.bounce_point = None
        self.post_impact_path = None
        self.last_fit_pred = None
        self.ball_state = BallState((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), False, 0.0)

        if BallKalmanFilter:
            self.kf = BallKalmanFilter(fps=30)
            self.ball_state = BallState((0.0, 0.0), (0.0, 0.0), (0.0, self.kf.gravity), False, 0.0)

        if PitchPlaneEstimator:
            self.pitch_estimator = PitchPlaneEstimator(
                warmup_frames=DETECTION_CONFIG['pitch_warmup_frames'],
                variance_threshold=DETECTION_CONFIG['pitch_variance_threshold'],
                min_static_ratio=DETECTION_CONFIG['pitch_min_static_ratio'],
                prefer_lower_half=DETECTION_CONFIG['pitch_prefer_lower_half'],
                texture_threshold=DETECTION_CONFIG['pitch_texture_threshold'],
                edge_low=DETECTION_CONFIG['pitch_edge_low'],
                edge_high=DETECTION_CONFIG['pitch_edge_high']
            )
        
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def process_frame(self, frame, frame_idx, yolo_detector, batsman_box=None, wicket_line=None):
        self.batsman_box = batsman_box
        self.wicket_line = wicket_line
        processed_frame = frame
        if preprocess_frame:
            processed_frame, _ = preprocess_frame(frame)
        frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        self._ensure_kf_initialized(processed_frame)
        kf_pred_pos = None
        kf_velocity = (0.0, 0.0)
        if self.kf:
            px, py = self.kf.predict()
            kf_pred_pos = (px, py)
            full_state = self.kf.get_full_state()
            kf_velocity = (full_state[2], full_state[3])
        self._update_pitch_plane(frame)
        ball_info = None
        
        if self.state == STATE_BOOTSTRAP:
            candidates = self._run_yolo_detection(processed_frame, yolo_detector, frame_idx)
            # Filter for higher confidence during bootstrap
            valid_candidates = [c for c in candidates if c['conf'] >= DETECTION_CONFIG['bootstrap_conf_threshold']]
            state_source = "physics"
            candidate_conf = 0.0
            
            if valid_candidates:
                cand = max(valid_candidates, key=lambda c: c['conf'])
                box = cand['box']
                cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
                area = cand['area']
                
                # Check for motion if we have a previous point
                motion_valid = True
                if self.last_center:
                    dist = math.hypot(cx - self.last_center[0], cy - self.last_center[1])
                    if dist < DETECTION_CONFIG['min_bootstrap_motion']:
                        # Stationary object detected - reset bootstrap
                        self.consecutive_detections = 0
                        motion_valid = False
                
                if motion_valid:
                    self.last_center = (cx, cy)
                    self.last_box = box
                    self.consecutive_detections += 1
                    self.bootstrap_history.append({
                        'frame_idx': frame_idx,
                        'center': (cx, cy),
                        'area': area,
                        'conf': cand['conf']
                    })
                    self.bootstrap_history = [
                        h for h in self.bootstrap_history
                        if frame_idx - h['frame_idx'] <= DETECTION_CONFIG['bootstrap_frames'] + 2
                    ]
                    if self.kf:
                        if not self.kf_initialized:
                            self.kf.initialize(cx, cy)
                            self.kf_initialized = True
                        else:
                            self.kf.update(cx, cy)

                    bootstrap_ready = self._bootstrap_motion_consistent()
                    if self.consecutive_detections >= DETECTION_CONFIG['bootstrap_frames'] or bootstrap_ready:
                        self._initialize_tracking(frame, box)
                        self.state = STATE_TRACKING
                        self.confidence = 1.0
                    state_source = "yolo"
                    candidate_conf = cand['conf']
            else:
                self.consecutive_detections = 0
                self.last_center = None # Clear history on loss during bootstrap
                self.bootstrap_history = []
                self._update_confidence(0.0, used_measurement=False)

            if state_source == "yolo":
                self._update_confidence(candidate_conf, used_measurement=True)

            self._enforce_physics_constraints()
            self._refresh_ball_state()
            ball_info = self._build_ball_info(state_source)
            self.prev_frame_gray = frame_gray
            return ball_info

        elif self.state == STATE_TRACKING:
            if kf_pred_pos:
                kf_pred_pos = self._apply_pitch_constraints(kf_pred_pos[0], kf_pred_pos[1], frame_idx)
            if self.kf:
                self.velocity_history.append(kf_velocity[1])

            fit_pred = self._get_fit_prediction(frame_idx)
            self.last_fit_pred = fit_pred
            of_success, of_pos, of_qual = self._track_optical_flow(frame_gray)
            tld_success, tld_pos, tld_box = self._track_tld(frame)
            candidates = self._run_yolo_detection(processed_frame, yolo_detector, frame_idx)
            
            best_candidate = None
            if kf_pred_pos:
                gating_radius = DETECTION_CONFIG['max_drift_pixels'] + 0.5 * math.hypot(*kf_velocity)
                best_score = float('inf')
                for cand in candidates:
                    dist = math.hypot(cand['center'][0] - kf_pred_pos[0], cand['center'][1] - kf_pred_pos[1])
                    if dist <= gating_radius:
                        last_area = self.last_box[2] * self.last_box[3] if self.last_box else cand['area']
                        size_score = abs(cand['area'] - last_area) / last_area
                        score = dist * 0.6 + size_score * 20.0
                        if score < best_score:
                            best_score = score
                            best_candidate = cand

            candidate_pos = None
            source = "physics"
            candidate_conf = 0.0
            if best_candidate:
                bx, by, bw, bh = best_candidate['box']
                last_area = self.last_box[2] * self.last_box[3] if self.last_box else (bw * bh)
                ratio = (bw * bh) / last_area
                if 0.5 <= ratio <= 2.2:
                    candidate_pos = (int(bx + bw / 2), int(by + bh / 2))
                    source = "yolo"
                    candidate_conf = best_candidate['conf']
                    self.last_box = best_candidate['box']
                    if not self._is_candidate_consistent(candidate_pos, candidate_conf, fit_pred):
                        candidate_pos = None
                        source = "unknown"

            if not candidate_pos and of_success and of_qual > DETECTION_CONFIG['optical_flow_quality_threshold']:
                candidate_pos = of_pos
                source = "optical_flow"
                candidate_conf = 0.6
                if not self._is_candidate_consistent(candidate_pos, candidate_conf, fit_pred):
                    candidate_pos = None
                    source = "unknown"

            if not candidate_pos and tld_success and tld_pos is not None:
                candidate_pos = tld_pos
                source = "tld"
                candidate_conf = 0.5
                if tld_box is not None:
                    self.last_box = tld_box
                if not self._is_candidate_consistent(candidate_pos, candidate_conf, fit_pred):
                    candidate_pos = None
                    source = "unknown"
            
            if not candidate_pos and self.csrt:
                success, box = self.csrt.update(frame)
                if success:
                    candidate_pos = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
                    self.last_box = [int(v) for v in box]
                    source = "csrt"
                    candidate_conf = 0.55
                    if not self._is_candidate_consistent(candidate_pos, candidate_conf, fit_pred):
                        candidate_pos = None
                        source = "unknown"

            if not candidate_pos and fit_pred and self.frames_without_update <= DETECTION_CONFIG['max_fit_gap']:
                candidate_pos = fit_pred
                source = "guided_recovery"
                candidate_conf = 0.35

            if not candidate_pos and kf_pred_pos:
                if self.kf:
                    self.kf.relax_constraints()
                candidate_pos = kf_pred_pos
                source = "physics"
                candidate_conf = 0.25

            if candidate_pos and source in ["yolo", "optical_flow", "csrt"]:
                if not self._is_measurement_acceptable(candidate_pos, source, kf_pred_pos):
                    candidate_pos = None
                    source = "physics"
                elif fit_pred is not None:
                    fit_residual = math.hypot(candidate_pos[0] - fit_pred[0], candidate_pos[1] - fit_pred[1])
                    if fit_residual > DETECTION_CONFIG['fit_residual_threshold']:
                        candidate_pos = fit_pred
                        source = "guided_recovery"
                        candidate_conf = 0.3
            if candidate_pos and source == "tld":
                if not self._is_measurement_acceptable(candidate_pos, source, kf_pred_pos):
                    candidate_pos = None
                    source = "physics"

            if candidate_pos:
                self.last_center = candidate_pos
                self.full_track_history.append(candidate_pos)
                if self.kf:
                    should_update = source in ["yolo", "optical_flow", "csrt", "tld", "guided_recovery"]
                    if should_update:
                        noise_scale = self._measurement_noise_scale(source)
                        if noise_scale is not None:
                            self.kf.set_measurement_noise_scale(noise_scale)
                        self.kf.update(candidate_pos[0], candidate_pos[1])
                        if noise_scale is not None:
                            self.kf.set_measurement_noise_scale(1.0)
                    full_state = self.kf.get_full_state()
                    kf_velocity = (full_state[2], full_state[3])
                    cov = self.kf.kf.errorCovPost[:2, :2]
                    self.uncertainties.append(float(np.trace(cov)))
                else:
                    self.uncertainties.append(10.0)
                
                used_measurement = source in ["yolo", "optical_flow", "csrt", "tld"]
                self._update_confidence(candidate_conf, used_measurement)
                if used_measurement:
                    self.frames_without_measurement = 0
                else:
                    self.frames_without_measurement += 1

                self._check_impact(frame_idx, candidate_pos, used_measurement)
                self._check_post_impact_prediction()

                self._enforce_physics_constraints()
                self._refresh_ball_state()
                if source in ["yolo", "optical_flow", "csrt", "tld", "guided_recovery"]:
                    self._update_fit_history(frame_idx, candidate_pos)
                ball_info = self._build_ball_info(source)
                if used_measurement:
                    self.frames_without_update = 0
                else:
                    self.frames_without_update += 1
            else:
                self.frames_without_update += 1
                self._update_confidence(0.0, used_measurement=False)
                self.frames_without_measurement += 1
                if self.frames_without_update > DETECTION_CONFIG['max_coast_frames']:
                    self.state = STATE_LOST

                self._check_impact(frame_idx, self.last_center, used_measurement=False)
                self._check_post_impact_prediction()
                self._enforce_physics_constraints()
                self._refresh_ball_state()
                ball_info = self._build_ball_info("physics")

            if (self.confidence < DETECTION_CONFIG['min_confidence']
                or self.frames_without_measurement > DETECTION_CONFIG['physics_prediction_horizon']):
                self.state = STATE_LOST
                self.reset()
                return None
            
            self.prev_frame_gray = frame_gray
            return ball_info
            
        elif self.state == STATE_LOST:
            self.reset()
            return None

    def _ensure_kf_initialized(self, frame):
        if not self.kf or self.kf_initialized:
            return
        h, w = frame.shape[:2]
        cx, cy = float(w * 0.5), float(h * 0.5)
        self.kf.initialize(cx, cy)
        self.kf_initialized = True

    def _enforce_physics_constraints(self):
        if not self.kf or not self.kf_initialized:
            return
        full_state = self.kf.get_full_state()
        vx, vy, ay = full_state[2], full_state[3], full_state[5]
        violations = 0

        if vx < 0.0:
            vx = 0.0
            violations += 1

        if not self.bounce_detected and vy < 0.0:
            vy = 0.0
            violations += 1

        if abs(ay - float(self.kf.gravity)) > 1e-3:
            ay = float(self.kf.gravity)
            violations += 1

        if violations:
            self.physics_violations += violations
        elif self.physics_violations > 0:
            self.physics_violations -= 1

        if self.physics_violations >= DETECTION_CONFIG['max_physics_violations']:
            self.state = STATE_LOST
            self.confidence = 0.0

        self.kf.set_state(vx=vx, vy=vy, ay=ay)

    def _refresh_ball_state(self):
        if self.kf:
            full_state = self.kf.get_full_state()
            position = (full_state[0], full_state[1])
            velocity = (full_state[2], full_state[3])
            acceleration = (0.0, float(self.kf.gravity))
        else:
            position = self.last_center if self.last_center else (0.0, 0.0)
            velocity = (0.0, 0.0)
            acceleration = (0.0, 0.0)
        self.ball_state = BallState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            has_bounced=self.bounce_detected,
            confidence=self.confidence,
            impact_frame=self.impact_frame,
            impact_point=self.impact_point,
            would_hit_stumps=self.would_hit_stumps,
            hit_frame=self.hit_frame,
            hit_point=self.hit_point,
            bounce_point=self.bounce_point,
            post_impact_path=self.post_impact_path
        )
        self.prev_physics_pos = self.ball_state.position

    def _check_impact(self, frame_idx, pos, used_measurement):
        if self.impact_frame is not None or not self.batsman_box:
            return
        if pos is None:
            return
        if self._point_in_box(pos, self.batsman_box, DETECTION_CONFIG['impact_roi_margin']):
            impact_pos = self.last_fit_pred if self.last_fit_pred is not None else pos
            self.impact_frame = int(frame_idx)
            self.impact_point = (float(impact_pos[0]), float(impact_pos[1]))
            return
        if (not used_measurement and
            self.frames_without_measurement >= DETECTION_CONFIG['impact_no_measurement_frames'] and
            self.last_center is not None and
            self._point_in_box(self.last_center, self.batsman_box, DETECTION_CONFIG['impact_roi_margin'])):
            impact_pos = self.last_fit_pred if self.last_fit_pred is not None else self.last_center
            self.impact_frame = int(frame_idx)
            self.impact_point = (float(impact_pos[0]), float(impact_pos[1]))

    def _check_post_impact_prediction(self):
        if self.impact_frame is None or self.would_hit_stumps is not None:
            return
        if not self.wicket_line or not self.kf:
            return
        full_state = self.kf.get_full_state()
        pos = (float(full_state[0]), float(full_state[1]))
        vx, vy = float(full_state[2]), float(full_state[3])
        ay = float(self.kf.gravity)

        x_line = float(self.wicket_line.get("x", 0.0))
        y_top = float(self.wicket_line.get("y_top", 0.0))
        y_bottom = float(self.wicket_line.get("y_bottom", 0.0))
        if y_bottom < y_top:
            y_top, y_bottom = y_bottom, y_top

        prev = pos
        path = [pos]
        for i in range(1, DETECTION_CONFIG['post_impact_predict_frames'] + 1):
            t = float(i)
            x = pos[0] + vx * t
            y = pos[1] + vy * t + 0.5 * ay * t * t
            path.append((x, y))
            if (prev[0] - x_line) * (x - x_line) <= 0:
                y_at_line = prev[1] + (y - prev[1]) * ((x_line - prev[0]) / (x - prev[0] + 1e-6))
                if y_at_line >= y_top and y_at_line <= y_bottom:
                    self.would_hit_stumps = True
                    self.hit_frame = int(self.impact_frame + i)
                    self.hit_point = (float(x_line), float(y_at_line))
                    self.post_impact_path = path
                    return
            prev = (x, y)
        self.would_hit_stumps = False
        self.post_impact_path = path

    def _point_in_box(self, pos, box, margin):
        x, y = float(pos[0]), float(pos[1])
        bx, by, bw, bh = [float(v) for v in box]
        return (x >= bx - margin and x <= bx + bw + margin and
                y >= by - margin and y <= by + bh + margin)

    def _update_fit_history(self, frame_idx, pos):
        if self.bounce_detected:
            self.post_bounce_history.append((int(frame_idx), float(pos[0]), float(pos[1])))
            if len(self.post_bounce_history) > DETECTION_CONFIG['fit_window'] * 2:
                self.post_bounce_history = self.post_bounce_history[-DETECTION_CONFIG['fit_window'] * 2:]
        else:
            self.pre_bounce_history.append((int(frame_idx), float(pos[0]), float(pos[1])))
            if len(self.pre_bounce_history) > DETECTION_CONFIG['fit_window'] * 2:
                self.pre_bounce_history = self.pre_bounce_history[-DETECTION_CONFIG['fit_window'] * 2:]

    def _fit_segment(self, history):
        if len(history) < 5:
            return None
        frames = np.array([h[0] for h in history], dtype=np.float32)
        xs = np.array([h[1] for h in history], dtype=np.float32)
        ys = np.array([h[2] for h in history], dtype=np.float32)
        f0 = frames[0]
        t = (frames - f0).reshape(-1, 1)
        try:
            coef_x = np.polyfit(t[:, 0], xs, 1)
            coef_y = np.polyfit(t[:, 0], ys, 2)
        except Exception:
            return None
        return coef_x, coef_y, f0

    def _get_fit_prediction(self, frame_idx):
        history = self.post_bounce_history if self.bounce_detected else self.pre_bounce_history
        fit = self._fit_segment(history)
        if not fit:
            return None
        coef_x, coef_y, f0 = fit
        t = float(frame_idx - f0)
        pred_x = coef_x[0] * t + coef_x[1]
        pred_y = coef_y[0] * t * t + coef_y[1] * t + coef_y[2]
        return (float(pred_x), float(pred_y))

    def _estimate_bounce_frame_idx(self, frame_idx):
        history = self.pre_bounce_history
        fit = self._fit_segment(history)
        if not fit or self.pitch_y is None:
            return frame_idx
        coef_x, coef_y, f0 = fit
        a, b, c = coef_y[0], coef_y[1], coef_y[2] - float(self.pitch_y)
        if abs(a) < 1e-6:
            if abs(b) < 1e-6:
                return frame_idx
            t = -c / b
            return int(round(f0 + max(0.0, t)))
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return frame_idx
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t2 = (-b - sqrt_disc) / (2.0 * a)
        candidates = [t for t in [t1, t2] if t >= 0.0]
        if not candidates:
            return frame_idx
        t = min(candidates, key=lambda v: abs((f0 + v) - frame_idx))
        return int(round(f0 + t))

    def _measurement_noise_scale(self, source):
        if source == "yolo":
            return DETECTION_CONFIG['yolo_noise_scale']
        if source == "optical_flow":
            return DETECTION_CONFIG['optical_noise_scale']
        if source == "csrt":
            return DETECTION_CONFIG['csrt_noise_scale']
        if source == "tld":
            return DETECTION_CONFIG['tld_noise_scale']
        if source == "guided_recovery":
            return DETECTION_CONFIG['guided_noise_scale']
        return None

    def _is_measurement_acceptable(self, pos, source, kf_pred_pos):
        if self.pitch_y is not None and pos[1] > self.pitch_y + DETECTION_CONFIG['pitch_margin']:
            return False

        if self.last_center:
            vx = pos[0] - self.last_center[0]
            vy = pos[1] - self.last_center[1]
            if abs(vx) > DETECTION_CONFIG['max_velocity'] or abs(vy) > DETECTION_CONFIG['max_velocity']:
                return False
            if not self.bounce_detected and vy < -DETECTION_CONFIG['min_upward_velocity']:
                return False
            if vx < -1.0:
                return False

        if kf_pred_pos:
            dist = math.hypot(pos[0] - kf_pred_pos[0], pos[1] - kf_pred_pos[1])
            if source == "yolo":
                gate = DETECTION_CONFIG['yolo_gate_px']
            elif source == "optical_flow":
                gate = DETECTION_CONFIG['optical_gate_px']
            elif source == "tld":
                gate = DETECTION_CONFIG['tld_gate_px']
            else:
                gate = DETECTION_CONFIG['csrt_gate_px']
            if dist > gate:
                return False

        return True

    def _build_ball_info(self, source):
        box = None
        if self.last_box:
            w, h = self.last_box[2], self.last_box[3]
            box = [
                int(self.ball_state.position[0] - w / 2),
                int(self.ball_state.position[1] - h / 2),
                int(w),
                int(h)
            ]
        return {
            "box": box,
            "conf": float(self.confidence),
            "source": source,
            "velocity": self.ball_state.velocity,
            "bounce": self.bounce_detected and (self.bounce_frame_idx >= 0),
            "state": self.ball_state.to_dict(),
            "trajectory": [
                [float(p[0]), float(p[1])]
                for p in self.full_track_history[-DETECTION_CONFIG['fit_window'] * 2:]
            ]
        }

    def _update_pitch_plane(self, frame):
        if not self.pitch_estimator or self.pitch_ready:
            return
        pitch_y = self.pitch_estimator.add_frame(frame)
        if pitch_y is not None:
            self.pitch_y = pitch_y
            self.pitch_model = self.pitch_estimator.get_model()
            self.pitch_ready = True

    def _apply_pitch_constraints(self, px, py, frame_idx):
        if not self.kf or self.pitch_y is None:
            return (px, py)
        pitch_y = self.pitch_y
        if self.pitch_estimator:
            pitch_at_x = self.pitch_estimator.get_pitch_y_at_x(px)
            if pitch_at_x is not None:
                pitch_y = pitch_at_x
        full_state = self.kf.get_full_state()
        vx, vy = full_state[2], full_state[3]

        if self.impact_frame is not None:
            if py >= pitch_y:
                clamped_y = pitch_y - DETECTION_CONFIG['pitch_margin']
                self.kf.set_state(y=clamped_y, vy=min(vy, 0.0))
                return (px, clamped_y)
            return (px, py)

        crossed = False
        if self.prev_physics_pos is not None:
            crossed = self.prev_physics_pos[1] < pitch_y and py >= pitch_y

        if not self.bounce_detected and crossed:
            self.bounce_detected = True
            self.bounce_frame_idx = self._estimate_bounce_frame_idx(frame_idx)
            self.post_bounce_history = []
            self.bounce_point = (float(px), float(pitch_y))
            new_vy = -abs(vy) * DETECTION_CONFIG['bounce_damping']
            self.kf.set_state(y=pitch_y, vy=new_vy)
            self.kf.reset_constraints()
            return (px, pitch_y)

        if py >= pitch_y:
            clamped_y = pitch_y - DETECTION_CONFIG['pitch_margin']
            self.kf.set_state(y=clamped_y, vy=min(vy, 0.0))
            return (px, clamped_y)

        return (px, py)

    def _is_candidate_consistent(self, pos, conf, fit_pred):
        if self.pitch_y is not None and not self.bounce_detected:
            if pos[1] > self.pitch_y + DETECTION_CONFIG['pitch_margin']:
                return False
        if fit_pred:
            dist = math.hypot(pos[0] - fit_pred[0], pos[1] - fit_pred[1])
            if dist > DETECTION_CONFIG['fit_residual_threshold'] and conf < DETECTION_CONFIG['high_confidence']:
                return False
            if dist > DETECTION_CONFIG['fit_residual_threshold'] * 1.5:
                return False
        return True

    def _update_confidence(self, meas_conf, used_measurement):
        if used_measurement:
            blended = 0.7 * self.confidence + 0.3 * max(0.5, meas_conf)
            self.confidence = min(1.0, blended)
        else:
            self.confidence = max(0.0, self.confidence - DETECTION_CONFIG['confidence_decay'])

    def _bootstrap_motion_consistent(self):
        if len(self.bootstrap_history) < 3:
            return False
        history = sorted(self.bootstrap_history, key=lambda h: h['frame_idx'])
        vectors = []
        for i in range(1, len(history)):
            p0 = history[i - 1]['center']
            p1 = history[i]['center']
            dx, dy = (p1[0] - p0[0], p1[1] - p0[1])
            if math.hypot(dx, dy) >= DETECTION_CONFIG['min_bootstrap_motion']:
                vectors.append((dx, dy))
        if len(vectors) < 2:
            return False
        angles = [math.atan2(v[1], v[0]) for v in vectors]
        angle_spread = max(angles) - min(angles)
        if angle_spread > math.radians(45):
            return False
        areas = [h['area'] for h in history[-3:]]
        if min(areas) <= 0:
            return False
        ratio = max(areas) / min(areas)
        return ratio <= 2.0

    def _initialize_tracking(self, frame, box):
        try:
            if not self.csrt:
                self.csrt = cv2.TrackerCSRT_create()
            H, W = frame.shape[:2]
            bx, by, bw, bh = box
            bx = max(0, min(W-1, int(bx)))
            by = max(0, min(H-1, int(by)))
            bw = max(1, min(W-bx, int(bw)))
            bh = max(1, min(H-by, int(bh)))
            self.csrt.init(frame, (bx, by, bw, bh))
        except Exception:
            self.csrt = None

        try:
            if not self.tld:
                self.tld = cv2.TrackerTLD_create()
            self.tld.init(frame, (bx, by, bw, bh))
        except Exception:
            self.tld = None

    def _run_yolo_detection(self, frame, detector, frame_idx):
        if frame is None or frame.size == 0 or detector is None: return []
        try:
            detections = detector.detect(frame, conf=0.2, iou=0.45)
            candidates = []
            for det in detections:
                x, y, w, h, conf = det[:5]
                area = w * h
                if DETECTION_CONFIG['min_area'] <= area <= DETECTION_CONFIG['max_area']:
                    candidates.append({
                        'box': [int(x), int(y), int(w), int(h)],
                        'conf': float(conf),
                        'area': area,
                        'center': (x + w / 2.0, y + h / 2.0)
                    })
            return candidates
        except Exception:
            return []

    def _track_optical_flow(self, current_gray):
        if self.last_center is None or self.prev_frame_gray is None: return False, None, 0.0
        try:
            p0 = np.array([[self.last_center]], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_gray, current_gray, p0, None, **self.lk_params)
            if st[0] == 1:
                new_pos = (float(p1[0][0][0]), float(p1[0][0][1]))
                if math.hypot(new_pos[0]-self.last_center[0], new_pos[1]-self.last_center[1]) < 120:
                    return True, new_pos, 1.0
        except Exception:
            pass
        return False, None, 0.0

    def _track_tld(self, frame):
        if self.tld is None:
            return False, None, None
        try:
            ok, box = self.tld.update(frame)
            if not ok:
                return False, None, None
            x, y, w, h = [int(v) for v in box]
            if w <= 0 or h <= 0:
                return False, None, None
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
            return True, (cx, cy), [x, y, w, h]
        except Exception:
            return False, None, None

_hybrid_tracker = None
_yolo_detector = None

def get_hybrid_tracker():
    global _hybrid_tracker
    if _hybrid_tracker is None: _hybrid_tracker = HybridBallTracker()
    return _hybrid_tracker

def get_yolo_detector(weights_path=None):
    global _yolo_detector
    if _yolo_detector is None:
        default_weights = "weights/ball-yolov8s.pt"
        target = weights_path if weights_path else default_weights
        if YOLOBallDetector:
            _yolo_detector = YOLOBallDetector(target)
    return _yolo_detector

def detect_ball_on_frame(frame, yolo_weights=None, frame_idx=0, batsman_box=None, wicket_line=None, **kwargs):
    """
    Main API for external use. USES HYBRID STATE MACHINE.
    """
    tracker = get_hybrid_tracker()
    detector = get_yolo_detector(yolo_weights)
    ball_info = tracker.process_frame(frame, frame_idx, detector, batsman_box=batsman_box, wicket_line=wicket_line)
    
    if ball_info:
        box = ball_info.get('box')
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{ball_info['source']}", (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if fit_trajectory and len(tracker.full_track_history) > 3:
            res = fit_trajectory(tracker.full_track_history)
            if res:
                pts = np.array(res[0], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255, 120, 0), 2, cv2.LINE_AA)
                curr_idx = len(tracker.full_track_history) - 1
                if curr_idx < len(tracker.uncertainties):
                    rad = int(math.sqrt(tracker.uncertainties[curr_idx]) * 2.0)
                    cv2.circle(frame, (int(tracker.last_center[0]), int(tracker.last_center[1])), 
                               max(5, rad), (200, 200, 200), 1, cv2.LINE_AA)

    return frame, ball_info
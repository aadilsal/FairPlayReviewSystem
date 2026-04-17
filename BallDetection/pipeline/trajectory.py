import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class SegmentModel:
    """Coefficients for a single trajectory segment (x and y as functions of t)."""
    x_coeffs: np.ndarray  # [a, b, c] for at^2 + bt + c
    y_coeffs: np.ndarray  # [a, b, c] for at^2 + bt + c
    start_frame: int
    end_frame: int

    def get_initial_conditions(self):
        """
        Returns initial position, velocity, acceleration for segment start (for Kalman seeding).
        """
        t0 = self.start_frame
        # Quadratic: x(t) = a*t^2 + b*t + c
        # Velocity: dx/dt = 2a*t + b
        # Acceleration: d^2x/dt^2 = 2a
        x0 = np.polyval(self.x_coeffs, t0)
        y0 = np.polyval(self.y_coeffs, t0)
        vx0 = 2*self.x_coeffs[0]*t0 + self.x_coeffs[1]
        vy0 = 2*self.y_coeffs[0]*t0 + self.y_coeffs[1]
        ax0 = 2*self.x_coeffs[0]
        ay0 = 2*self.y_coeffs[0]
        return (np.array([x0, y0]), np.array([vx0, vy0]), np.array([ax0, ay0]))

@dataclass
class TrajectoryModel:
    """Complete trajectory model with multiple segments (pre- and post-bounce)."""
    segments: List[SegmentModel]
    bounce_frame: Optional[int] = None
    bounce_time: Optional[float] = None


def _estimate_bounce_time(sorted_anchors: List[Dict[str, Any]], flip_idx: int) -> Optional[float]:
    """
    Estimate the visual bounce time from three anchor points around the sign flip.
    Uses a quadratic fit on y(t) and returns the vertex time when possible.
    """
    if flip_idx < 0 or flip_idx + 2 >= len(sorted_anchors):
        return None

    window = sorted_anchors[flip_idx:flip_idx + 3]
    t = np.array([a['frame_idx'] for a in window], dtype=np.float64)
    y = np.array([a['interpolated_position'][1] for a in window], dtype=np.float64)

    if len(np.unique(t)) < 3:
        return float(t[1])

    try:
        coeffs = np.polyfit(t, y, 2)
    except Exception:
        return float(t[1])

    a, b, _c = coeffs
    if abs(a) < 1e-9:
        return float(t[1])

    bounce_t = -b / (2.0 * a)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    return float(np.clip(bounce_t, t_min, t_max))

def find_bounce_frame(anchors: List[Dict[str, Any]]) -> Optional[int]:
    """
    Detects the frame where vertical velocity vy flips sign (downward to upward).
    In image coordinates, y increases downwards, so a bounce is a flip from +vy to -vy.
    """
    if len(anchors) < 3:
        return None

    # Sort anchors by frame index just in case
    sorted_anchors = sorted(anchors, key=lambda x: x['frame_idx'])
    
    velocities = []
    for i in range(len(sorted_anchors) - 1):
        a1 = sorted_anchors[i]
        a2 = sorted_anchors[i+1]
        
        dt = a2['frame_idx'] - a1['frame_idx']
        if dt == 0:
            continue
            
        dy = a2['interpolated_position'][1] - a1['interpolated_position'][1]
        vy = dy / dt
        velocities.append((a1['frame_idx'], vy))

    # Look for a significant flip from positive (falling) to negative (rising)
    for i in range(len(velocities) - 1):
        t1, v1 = velocities[i]
        t2, v2 = velocities[i+1]
        
        # In image coords: v > 0 is downward, v < 0 is upward
        if v1 > 2.0 and v2 < -2.0:  # Use a threshold to avoid noise
            bounce_time = _estimate_bounce_time(sorted_anchors, i)
            if bounce_time is not None:
                bounce_frame = int(np.floor(bounce_time))
                logger.info(
                    f"[TRAJECTORY] Detected bounce near frame {bounce_frame} (t={bounce_time:.2f})"
                )
                return bounce_frame

            # Fallback: use the middle anchor when we cannot refine the vertex.
            bounce_frame = sorted_anchors[i+1]['frame_idx']
            logger.info(f"[TRAJECTORY] Detected bounce near frame {bounce_frame}")
            return bounce_frame
            
    return None


def find_bounce_event(anchors: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[float]]:
    """Return both the display frame and fractional bounce time when a bounce is detected."""
    if len(anchors) < 3:
        return None, None

    sorted_anchors = sorted(anchors, key=lambda x: x['frame_idx'])

    velocities = []
    for i in range(len(sorted_anchors) - 1):
        a1 = sorted_anchors[i]
        a2 = sorted_anchors[i + 1]
        dt = a2['frame_idx'] - a1['frame_idx']
        if dt == 0:
            continue
        dy = a2['interpolated_position'][1] - a1['interpolated_position'][1]
        vy = dy / dt
        velocities.append((a1['frame_idx'], vy))

    for i in range(len(velocities) - 1):
        _t1, v1 = velocities[i]
        _t2, v2 = velocities[i + 1]
        if v1 > 2.0 and v2 < -2.0:
            bounce_time = _estimate_bounce_time(sorted_anchors, i)
            if bounce_time is not None:
                return int(np.floor(bounce_time)), bounce_time
            return sorted_anchors[i + 1]['frame_idx'], float(sorted_anchors[i + 1]['frame_idx'])

    return None, None

def fit_trajectory(anchors: List[Dict[str, Any]]) -> TrajectoryModel:
    """
    Fits degree-2 polynomials to anchor positions.
    Splits into pre- and post-bounce segments if a bounce is detected.
    """
    if not anchors:
        return TrajectoryModel(segments=[])

    bounce_frame, bounce_time = find_bounce_event(anchors)
    
    if bounce_frame is None:
        # Fit a single segment
        segment = _fit_segment(anchors)
        return TrajectoryModel(segments=[segment] if segment else [])
    
    # Split anchors based on bounce
    split_t = bounce_time if bounce_time is not None else float(bounce_frame)
    pre_anchors = [a for a in anchors if a['frame_idx'] < split_t]
    post_anchors = [a for a in anchors if a['frame_idx'] >= split_t]
    
    segments = []
    seg_pre = _fit_segment(pre_anchors)
    if seg_pre:
        segments.append(seg_pre)
        
    seg_post = _fit_segment(post_anchors)
    if seg_post:
        segments.append(seg_post)
        
    return TrajectoryModel(segments=segments, bounce_frame=bounce_frame, bounce_time=bounce_time)

def _fit_segment(anchors: List[Dict[str, Any]]) -> Optional[SegmentModel]:
    """Helper to fit a single degree-2 polynomial segment."""
    if len(anchors) < 3:
        # Not enough points for a quadratic fit, try linear if possible
        if len(anchors) < 2:
            return None
        deg = 1
    else:
        deg = 2

    t = np.array([a['frame_idx'] for a in anchors])
    x = np.array([a['interpolated_position'][0] for a in anchors])
    y = np.array([a['interpolated_position'][1] for a in anchors])

    x_coeffs = np.polyfit(t, x, deg)
    y_coeffs = np.polyfit(t, y, deg)

    # Pad with zeros if deg=1 to keep array size consistent (3 elements for deg=2)
    if deg == 1:
        x_coeffs = np.insert(x_coeffs, 0, 0)
        y_coeffs = np.insert(y_coeffs, 0, 0)

    segment = SegmentModel(
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        start_frame=int(t[0]),
        end_frame=int(t[-1])
    )
    # segment.get_initial_conditions() can be used for Kalman seeding
    return segment

def predict_position(model: TrajectoryModel, frame_idx: int) -> Optional[Tuple[float, float]]:
    """
    Returns the predicted (x, y) position for a given frame index.
    Selects the appropriate segment if multiple exist.
    """
    if not model.segments:
        return None
        
    # Choose segment: 
    # If frame is before bounce, use first segment. 
    # If after bounce, use the last segment.
    target_segment = model.segments[0]
    bounce_split = model.bounce_time if model.bounce_time is not None else model.bounce_frame
    if bounce_split is not None and frame_idx >= bounce_split:
        target_segment = model.segments[-1]
    
    # We can also be more precise and pick based on start/end if there are more than 2 segments
    for seg in model.segments:
        if seg.start_frame <= frame_idx <= seg.end_frame:
            target_segment = seg
            break
            
    x = np.polyval(target_segment.x_coeffs, frame_idx)
    y = np.polyval(target_segment.y_coeffs, frame_idx)
    
    return (float(x), float(y))
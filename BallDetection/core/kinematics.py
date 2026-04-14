import numpy as np
import logging
from typing import Tuple, List, Optional
from BallDetection.pipeline.trajectory import TrajectoryModel, predict_position

logger = logging.getLogger(__name__)

def project_position(trajectory_model: TrajectoryModel, frame_idx: int) -> Optional[Tuple[float, float]]:
    """
    Pure parabolic projection from the fitted trajectory — the last resort.
    Wraps the trajectory model's prediction function. The caller should tag 
    the result with source: 'kinematic'.
    """
    return predict_position(trajectory_model, frame_idx)

def find_edge_intersection(forward_model, backward_model, gap_range: Optional[Tuple[int, int]] = None) -> Optional[Tuple[float, float]]:
    """
    Resolves an occlusion edge using the available forward/backward track samples.

    The current post-processing layer passes point lists, so this helper accepts
    either sequences of (x, y) samples or model-like objects with evaluate().
    """

    def _as_point_sequence(source, prefer_last: bool) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []

        if source is None:
            return points

        if isinstance(source, (list, tuple, np.ndarray)):
            for item in source:
                if item is None or not isinstance(item, (list, tuple, np.ndarray)):
                    continue
                if len(item) < 2:
                    continue
                points.append((float(item[0]), float(item[1])))
            return points

        if hasattr(source, "evaluate") and gap_range is not None:
            start_frame, end_frame = gap_range
            t_star = (start_frame + end_frame) / 2.0
            x_val, y_val = source.evaluate(t_star)
            return [(float(x_val), float(y_val))]

        if hasattr(source, "x_coeffs") and hasattr(source, "y_coeffs") and gap_range is not None:
            start_frame, end_frame = gap_range
            t_star = (start_frame + end_frame) / 2.0
            x_val = np.polyval(source.x_coeffs, t_star)
            y_val = np.polyval(source.y_coeffs, t_star)
            return [(float(x_val), float(y_val))]

        return points

    forward_points = _as_point_sequence(forward_model, prefer_last=True)
    backward_points = _as_point_sequence(backward_model, prefer_last=False)

    if not forward_points or not backward_points:
        return None

    fx, fy = forward_points[-1]
    bx, by = backward_points[0]
    return ((fx + bx) / 2.0, (fy + by) / 2.0)

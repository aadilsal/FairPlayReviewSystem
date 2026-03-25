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

def find_edge_intersection(forward_model, backward_model, gap_range: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """
    Finds the intersection of two parabolic arcs (quadratic segment models) for occlusion gaps.
    Args:
        forward_model: SegmentModel for the forward arc (with .a, .b, .c coefficients)
        backward_model: SegmentModel for the backward arc (with .a, .b, .c coefficients)
        gap_range: (start_frame, end_frame) tuple for the occlusion gap
    Returns:
        (x, y) intersection point, or midpoint average if no real intersection.
    """
    # Extract quadratic coefficients
    a1, b1, c1 = forward_model.a, forward_model.b, forward_model.c
    a2, b2, c2 = backward_model.a, backward_model.b, backward_model.c
    start_frame, end_frame = gap_range
    gap_mid = (start_frame + end_frame) / 2

    # Solve (a1-a2)t^2 + (b1-b2)t + (c1-c2) = 0
    coeffs = [a1 - a2, b1 - b2, c1 - c2]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r)]

    # Select root(s) within gap range
    valid_roots = [t for t in real_roots if start_frame <= t <= end_frame]

    if not valid_roots:
        # No intersection: fallback to midpoint average
        logger.warning("[KINEMATICS] No real intersection found, using midpoint average.")
        x1, y1 = forward_model.evaluate(gap_mid)
        x2, y2 = backward_model.evaluate(gap_mid)
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    # If multiple valid roots, pick closest to gap midpoint
    t_star = min(valid_roots, key=lambda t: abs(t - gap_mid))

    # Evaluate both models at t_star and average
    x1, y1 = forward_model.evaluate(t_star)
    x2, y2 = backward_model.evaluate(t_star)
    return ((x1 + x2) / 2, (y1 + y2) / 2)

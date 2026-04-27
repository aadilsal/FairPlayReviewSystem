"""Trajectory math helpers used by LBW analysis."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("fairplay.trajectory_analyzer")


def _sample_segment(segment: Any) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    start_t = float(segment.start_frame)
    end_t = float(segment.end_frame)
    for t in np.arange(start_t, end_t + 1.0, 1.0):
        x = float(np.polyval(segment.x_coeffs, t))
        y = float(np.polyval(segment.y_coeffs, t))
        points.append((t, x, y))
    return points


def get_trajectory_polyline(trajectory_model: Any) -> List[Tuple[float, float]]:
    polyline: List[Tuple[float, float]] = []
    if not trajectory_model or not getattr(trajectory_model, "segments", None):
        return polyline
    for segment in trajectory_model.segments:
        for _t, x, y in _sample_segment(segment):
            polyline.append((x, y))
    return polyline


def split_fitted_polyline_by_bounce(
    trajectory_model: Any,
    bounce_frame: Optional[int],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], int]:
    pre: List[Tuple[float, float]] = []
    post: List[Tuple[float, float]] = []
    fitted_start_frame = 0
    if not trajectory_model or not getattr(trajectory_model, "segments", None):
        return pre, post, fitted_start_frame

    fitted_start_frame = int(trajectory_model.segments[0].start_frame)
    split_frame = int(bounce_frame) if bounce_frame is not None else None

    for segment in trajectory_model.segments:
        for t, x, y in _sample_segment(segment):
            if split_frame is None:
                pre.append((x, y))
            elif int(round(t)) < split_frame:
                pre.append((x, y))
            else:
                post.append((x, y))

    if split_frame is not None and not pre and post:
        pre.append(post[0])
    return pre, post, fitted_start_frame


def _estimate_velocity(pre_impact_anchors: List[dict]) -> Tuple[float, float]:
    if len(pre_impact_anchors) < 2:
        return 0.0, 0.0

    tail = pre_impact_anchors[-5:]
    t = np.array([float(a["frame_idx"]) for a in tail], dtype=np.float64)
    x = np.array([float(a["interpolated_position"][0]) for a in tail], dtype=np.float64)
    y = np.array([float(a["interpolated_position"][1]) for a in tail], dtype=np.float64)

    if np.ptp(t) < 1e-6:
        return 0.0, 0.0

    vx = float(np.polyfit(t, x, 1)[0])
    vy = float(np.polyfit(t, y, 1)[0])
    return vx, vy


def project_from_impact(
    pre_impact_anchors: List[dict],
    impact_point: Tuple[float, float],
    frames_ahead: int = 30,
    gravity_px_per_f2: float = 0.35,
) -> List[Tuple[float, float]]:
    projected: List[Tuple[float, float]] = []
    if frames_ahead <= 0:
        return projected

    vx, vy = _estimate_velocity(pre_impact_anchors)
    x0 = float(impact_point[0])
    y0 = float(impact_point[1])

    for step in range(1, frames_ahead + 1):
        dt = float(step)
        x = x0 + vx * dt
        y = y0 + vy * dt + 0.5 * gravity_px_per_f2 * dt * dt
        projected.append((float(x), float(y)))
    return projected


def find_wicket_intersection(
    projected_polyline: List[Tuple[float, float]],
    wicket_line: Tuple[float, float, float, float],
    wicket_height_px: float = 65.0,
    x_tolerance_px: float = 8.0,
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    if not projected_polyline:
        return False, None

    x_near, y_base, x_far, _y_unused = wicket_line
    x_min = min(float(x_near), float(x_far)) - float(x_tolerance_px)
    x_max = max(float(x_near), float(x_far)) + float(x_tolerance_px)
    y_base = float(y_base)
    y_top = y_base - float(wicket_height_px)

    for x, y in projected_polyline:
        if x_min <= x <= x_max and y_top <= y <= y_base:
            return True, (float(x), float(y))
    return False, None


def extend_trajectory(trajectory_model: Any, num_frames: int = 30) -> List[Tuple[float, float]]:
    # Legacy wrapper for compatibility inside this module.
    if not trajectory_model or not getattr(trajectory_model, "segments", None):
        return []
    tail_segment = trajectory_model.segments[-1]
    anchors = [
        {
            "frame_idx": int(tail_segment.start_frame),
            "interpolated_position": (
                float(np.polyval(tail_segment.x_coeffs, tail_segment.start_frame)),
                float(np.polyval(tail_segment.y_coeffs, tail_segment.start_frame)),
            ),
        },
        {
            "frame_idx": int(tail_segment.end_frame),
            "interpolated_position": (
                float(np.polyval(tail_segment.x_coeffs, tail_segment.end_frame)),
                float(np.polyval(tail_segment.y_coeffs, tail_segment.end_frame)),
            ),
        },
    ]
    impact = anchors[-1]["interpolated_position"]
    return project_from_impact(anchors, impact, frames_ahead=num_frames, gravity_px_per_f2=0.0)


def check_wicket_intersect(
    trajectory_model: Any,
    wicket_x_near: float,
    wicket_x_far: float,
    wicket_y_base: float,
    stump_height: float = 50.0,
    tolerance_x: float = 20.0,
) -> Tuple[bool, Optional[str]]:
    polyline = get_trajectory_polyline(trajectory_model)
    hit, point = find_wicket_intersection(
        polyline,
        (wicket_x_near, wicket_y_base, wicket_x_far, wicket_y_base),
        wicket_height_px=stump_height,
        x_tolerance_px=tolerance_x,
    )
    if not hit or point is None:
        return False, None
    return True, f"trajectory_intersects_wickets_x{point[0]:.1f}_y{point[1]:.1f}"


def get_trajectory_segment_range(trajectory_model: Any) -> Optional[Tuple[int, int]]:
    if not trajectory_model or not getattr(trajectory_model, "segments", None):
        return None
    start = int(trajectory_model.segments[0].start_frame)
    end = int(trajectory_model.segments[-1].end_frame)
    return start, end

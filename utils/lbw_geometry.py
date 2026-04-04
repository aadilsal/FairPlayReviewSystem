"""
2D image-space wicket axis for LBW inline checks and stump-plane extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WicketGeometry:
    """Far→near wicket axis in pixel coordinates."""

    far_center: Tuple[float, float]
    near_center: Tuple[float, float]
    u: np.ndarray  # unit vector far → near, shape (2,)
    near_box: Tuple[float, float, float, float]  # x, y, w, h
    far_box: Tuple[float, float, float, float]
    lateral_threshold: float  # max perpendicular distance for "inline" / hitting stumps (px)
    stump_y_top: float
    stump_y_bottom: float

    @property
    def s_stump(self) -> float:
        """Projection scalar of striker stumps along the wicket line from far_center."""
        return self.projection_s(self.near_center[0], self.near_center[1])

    def projection_s(self, px: float, py: float) -> float:
        """Signed distance along u from far_center to foot of perpendicular from (px, py)."""
        p = np.array([px, py], dtype=np.float64)
        f = np.array(self.far_center, dtype=np.float64)
        return float(np.dot(p - f, self.u))

    def inline_distance(self, px: float, py: float) -> float:
        """Perpendicular distance from point to the infinite line through far–near centers."""
        p = np.array([px, py], dtype=np.float64)
        f = np.array(self.far_center, dtype=np.float64)
        n = np.array([-self.u[1], self.u[0]], dtype=np.float64)
        return abs(float(np.dot(p - f, n)))

    def extended_line_segment(
        self, extend_near: float = 120.0, extend_far: float = 80.0
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Endpoints for drawing the wicket line (extended past both stumps)."""
        f = np.array(self.far_center, dtype=np.float64)
        n = np.array(self.near_center, dtype=np.float64)
        p0 = f - self.u * extend_far
        p1 = n + self.u * extend_near
        return (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))


def parse_far_near_boxes(det_wickets: List[Dict[str, Any]]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    far_b, near_b = None, None
    for w in det_wickets or []:
        lbl = w.get("label", "")
        box = w.get("box")
        if not box or len(box) != 4:
            continue
        if "Far" in lbl:
            far_b = [float(x) for x in box]
        elif "Near" in lbl:
            near_b = [float(x) for x in box]
    return far_b, near_b


def wicket_geometry_from_boxes(
    far_box: List[float], near_box: List[float], lateral_scale: float = 0.45
) -> WicketGeometry:
    fx, fy, fw, fh = far_box
    nx, ny, nw, nh = near_box
    far_c = (fx + fw / 2.0, fy + fh / 2.0)
    near_c = (nx + nw / 2.0, ny + nh / 2.0)
    d = np.array([near_c[0] - far_c[0], near_c[1] - far_c[1]], dtype=np.float64)
    norm = float(np.linalg.norm(d)) + 1e-9
    u = d / norm
    lateral_threshold = max(10.0, nw * lateral_scale)
    return WicketGeometry(
        far_center=far_c,
        near_center=near_c,
        u=u,
        near_box=tuple(near_box),
        far_box=tuple(far_box),
        lateral_threshold=lateral_threshold,
        stump_y_top=float(ny),
        stump_y_bottom=float(ny + nh),
    )


def pick_reference_wicket_boxes(
    wickets_per_frame: List[List[Dict[str, Any]]],
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Use the last frame that has both far and near wickets (stable end of clip)."""
    far_b, near_b = None, None
    for dets in wickets_per_frame:
        f, n = parse_far_near_boxes(dets)
        if f is not None:
            far_b = f
        if n is not None:
            near_b = n
        if f is not None and n is not None:
            far_b, near_b = f, n
    return far_b, near_b

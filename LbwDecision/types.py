from __future__ import annotations

from typing import Optional, Tuple, TypedDict, List


Point2D = Tuple[float, float]


class Anchor(TypedDict):
    frame_idx: int
    interpolated_position: Point2D


class EventPoint(TypedDict):
    frame_idx: int
    point: Point2D


class LbwOverlayResult(TypedDict, total=False):
    pitch_inline: bool
    impact_inline: bool
    pad_contact: bool
    wickets_hitting: bool
    pitch_point: Optional[Point2D]
    impact_point: Optional[Point2D]
    bounce_frame: Optional[int]
    pitch_frame_idx: Optional[int]
    impact_frame_idx: Optional[int]
    stump_intersection: Optional[Point2D]
    fitted_polyline: List[Point2D]
    predicted_extension: List[Point2D]
    pre_bounce_polyline: List[Point2D]
    post_bounce_polyline: List[Point2D]
    projected_from_impact_polyline: List[Point2D]
    wicket_line: Optional[List[List[float]]]
    decision: str
    geometric_lbw: bool
    reason: Optional[str]
    has_valid_pitch: bool
    has_valid_impact: bool

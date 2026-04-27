"""LBW analysis facade with compatibility-safe outputs for pipeline/API consumers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from .anchor_utils import (
        build_anchors_from_ball_infos as _build_anchors,
        index_anchors_by_frame,
        slice_anchors_before_frame,
    )
    from .bounce_line_validator import is_impact_in_line, is_pitch_in_line
    from .events import find_first_bounce_event, find_first_contact_after_pitch
    from .trajectory_analyzer import (
        find_wicket_intersection,
        get_trajectory_polyline,
        project_from_impact,
        split_fitted_polyline_by_bounce,
    )
    from .wicket_geometry import resolve_wicket_positions
except ImportError:
    from anchor_utils import (  # type: ignore
        build_anchors_from_ball_infos as _build_anchors,
        index_anchors_by_frame,
        slice_anchors_before_frame,
    )
    from bounce_line_validator import is_impact_in_line, is_pitch_in_line  # type: ignore
    from events import find_first_bounce_event, find_first_contact_after_pitch  # type: ignore
    from trajectory_analyzer import (  # type: ignore
        find_wicket_intersection,
        get_trajectory_polyline,
        project_from_impact,
        split_fitted_polyline_by_bounce,
    )
    from wicket_geometry import resolve_wicket_positions  # type: ignore

logger = logging.getLogger("fairplay.lbw_analyzer")


def _build_default_result(reason: Optional[str], bounce_frame: Optional[int]) -> Dict[str, Any]:
    return {
        "pitch_inline": False,
        "impact_inline": False,
        "pad_contact": False,
        "bat_contact": False,
        "impact_target": None,
        "wickets_hitting": False,
        "pitch_point": None,
        "impact_point": None,
        "bounce_frame": bounce_frame,
        "pitch_frame_idx": None,
        "impact_frame_idx": None,
        "stump_intersection": None,
        "fitted_polyline": [],
        "predicted_extension": [],
        "pre_bounce_polyline": [],
        "post_bounce_polyline": [],
        "projected_from_impact_polyline": [],
        "wicket_line": None,
        "decision": "NOT OUT",
        "geometric_lbw": False,
        "reason": reason,
        "has_valid_pitch": False,
        "has_valid_impact": False,
    }


def _derive_decision(
    has_valid_pitch: bool,
    has_valid_impact: bool,
    wickets_hitting: bool,
) -> Tuple[str, str, bool]:
    if has_valid_pitch and has_valid_impact and wickets_hitting:
        return "OUT", "All LBW geometric checks passed", True
    if not has_valid_pitch:
        return "NOT OUT", "No valid pitch point", False
    if not has_valid_impact:
        return "NOT OUT", "No valid impact point", False
    return "NOT OUT", "Projected trajectory misses wickets", False


def build_anchors_from_ball_infos(ball_infos: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # Public API retained for pipeline/API compatibility.
    return _build_anchors(ball_infos)


def analyze_lbw_sequence(
    ball_infos: List[Optional[Dict[str, Any]]],
    trajectory_model: Any,
    wickets_per_frame: List[List[Dict[str, Any]]],
    pads_per_frame: List[List[Dict[str, Any]]],
    batsman_boxes: List[Optional[List[int]]],
    bats_per_frame: Optional[List[List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    # batsman_boxes is kept in signature for compatibility, currently not needed here.
    _ = batsman_boxes

    bounce_hint = None
    if trajectory_model is not None:
        bounce_hint = getattr(trajectory_model, "bounce_frame", None)
    result = _build_default_result(reason=None, bounce_frame=bounce_hint)

    anchors = _build_anchors(ball_infos)
    if not anchors:
        result["reason"] = "No ball detections available"
        return result

    pre_bounce, post_bounce, fitted_start_frame = split_fitted_polyline_by_bounce(
        trajectory_model, bounce_hint
    )
    fitted_polyline = get_trajectory_polyline(trajectory_model)
    result["pre_bounce_polyline"] = pre_bounce
    result["post_bounce_polyline"] = post_bounce
    result["fitted_polyline"] = fitted_polyline
    result["fitted_start_frame"] = fitted_start_frame

    wicket_line = resolve_wicket_positions(wickets_per_frame)
    wicket_x_near = wicket_y_base = wicket_x_far = None
    if wicket_line:
        wicket_x_near, wicket_y_base, wicket_x_far, _wicket_y_unused = wicket_line
        result["wicket_line"] = [
            [float(wicket_x_near), float(wicket_y_base)],
            [float(wicket_x_far), float(wicket_y_base)],
        ]

    pitch_event = find_first_bounce_event(anchors, bounce_hint_frame=bounce_hint)
    if pitch_event is not None:
        result["has_valid_pitch"] = True
        result["pitch_frame_idx"] = int(pitch_event["frame_idx"])
        result["pitch_point"] = (
            float(pitch_event["point"][0]),
            float(pitch_event["point"][1]),
        )
        result["bounce_frame"] = int(pitch_event["frame_idx"])
        if wicket_line:
            result["pitch_inline"] = is_pitch_in_line(
                float(pitch_event["point"][0]),
                float(wicket_x_near),
                float(wicket_x_far),
            )

    anchors_by_frame = index_anchors_by_frame(anchors)
    if result["has_valid_pitch"]:
        impact_event = find_first_contact_after_pitch(
            anchors_by_frame,
            pads_per_frame,
            bats_per_frame,
            int(result["pitch_frame_idx"]),
            overlap_margin=0.0,
        )
        if impact_event is not None:
            result["has_valid_impact"] = True
            result["impact_frame_idx"] = int(impact_event["frame_idx"])
            result["impact_point"] = (
                float(impact_event["point"][0]),
                float(impact_event["point"][1]),
            )
            result["impact_target"] = str(impact_event.get("target") or "pad")
            result["pad_contact"] = result["impact_target"] == "pad"
            result["bat_contact"] = result["impact_target"] == "bat"
            if wicket_line:
                result["impact_inline"] = is_impact_in_line(
                    float(impact_event["point"][0]),
                    float(wicket_x_near),
                    float(wicket_x_far),
                )

    projected_from_impact: List[Tuple[float, float]] = []
    if (
        result["has_valid_impact"]
        and result["impact_point"] is not None
        and result.get("pad_contact", False)
        and not result.get("bat_contact", False)
    ):
        pre_impact_anchors = slice_anchors_before_frame(
            anchors,
            int(result["impact_frame_idx"]),
            min_points=3,
        )
        projected_from_impact = project_from_impact(
            pre_impact_anchors,
            result["impact_point"],
            frames_ahead=30,
            gravity_px_per_f2=0.35,
        )
        if wicket_line:
            hit, hit_point = find_wicket_intersection(
                projected_from_impact,
                wicket_line,
                wicket_height_px=65.0,
                x_tolerance_px=8.0,
            )
            result["wickets_hitting"] = bool(hit)
            result["stump_intersection"] = hit_point

    result["projected_from_impact_polyline"] = projected_from_impact
    # Keep compatibility with current visualizer keys.
    result["predicted_extension"] = projected_from_impact

    decision, reason, geometric_lbw = _derive_decision(
        bool(result["has_valid_pitch"]),
        bool(result["has_valid_impact"]),
        bool(result["wickets_hitting"]),
    )
    if result.get("bat_contact", False):
        decision = "NOT OUT"
        geometric_lbw = False
        reason = "Bat contact detected"
    if not wicket_line and bool(result["has_valid_pitch"]) and bool(result["has_valid_impact"]):
        decision = "NOT OUT"
        geometric_lbw = False
        if not result.get("bat_contact", False):
            reason = "Wickets not detected"

    result["decision"] = decision
    result["reason"] = reason
    result["geometric_lbw"] = geometric_lbw

    return result


def lbw_overlay_for_api(lbw_overlay: Dict[str, Any]) -> Dict[str, Any]:
    # Preserve existing API field names and value vocabulary.
    impact_str = "hitting" if lbw_overlay.get("has_valid_impact") else "missing"
    pitch_str = "in-line" if lbw_overlay.get("pitch_inline") else "outside"
    wickets_str = "hitting" if lbw_overlay.get("wickets_hitting") else "missing"

    return {
        "impact": impact_str,
        "pitch": pitch_str,
        "wickets": wickets_str,
        "decision": lbw_overlay.get("decision", "NOT OUT"),
        "reason": lbw_overlay.get("reason"),
    }

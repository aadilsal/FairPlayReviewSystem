"""
Geometric LBW analysis: pitch inline, pad/body impact inline, predicted stump hit.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from BallDetection.pipeline.trajectory import TrajectoryModel, predict_position, SegmentModel

from lbw_geometry import WicketGeometry, pick_reference_wicket_boxes, wicket_geometry_from_boxes

# Image y grows downward; post-bounce ball should accelerate downward (pixels / frame^2).
_DEFAULT_GRAVITY_Y = 0.42

# When required data is absent, do not imply NOT OUT — use NO_DECISION + reason "Missing …".
NO_DECISION = "NO_DECISION"


def _mark_missing_data(out: Dict[str, Any], data_label: str) -> None:
    out["decision"] = NO_DECISION
    out["reason"] = f"Missing {data_label}"
    out["geometric_lbw"] = False


def missing_clip_overlay(n_frames: int = 0) -> Dict[str, Any]:
    """Placeholder when no frames were processed."""
    o = _empty_overlay(n_frames)
    _mark_missing_data(o, "frames")
    return o


def _segment_for_time(model: TrajectoryModel, t: float) -> Optional[SegmentModel]:
    if not model.segments:
        return None
    target = model.segments[0]
    bounce_split = model.bounce_time if getattr(model, "bounce_time", None) is not None else model.bounce_frame
    if bounce_split is not None and t >= float(bounce_split):
        target = model.segments[-1]
    for seg in model.segments:
        if seg.start_frame <= t <= seg.end_frame:
            target = seg
            break
    return target


def sample_ballistic_extension(
    model: TrajectoryModel,
    t_start: float,
    geom: Optional[WicketGeometry] = None,
    stop_on_reverse: bool = False,
    num_steps: int = 260,
    dt: float = 1.0,
    gravity_y: float = _DEFAULT_GRAVITY_Y,
    min_projection_step: float = 1e-4,
) -> List[Tuple[float, float]]:
    """
    Continue the path past the last fit using velocity at t_start plus constant
    downward acceleration in image space. Avoids evaluating the quadratic far past
    the data, which bends back after the parabola vertex.
    """
    seg = _segment_for_time(model, t_start)
    if seg is None:
        return []
    x = float(np.polyval(seg.x_coeffs, t_start))
    y = float(np.polyval(seg.y_coeffs, t_start))
    vx = float(2.0 * seg.x_coeffs[0] * t_start + seg.x_coeffs[1])
    vy = float(2.0 * seg.y_coeffs[0] * t_start + seg.y_coeffs[1])
    pts: List[Tuple[float, float]] = []
    prev_s: Optional[float] = None
    toward_sign = 0.0
    if geom is not None and stop_on_reverse:
        prev_s = geom.projection_s(x, y)
        toward_delta = geom.s_stump - prev_s
        if toward_delta > 0:
            toward_sign = 1.0
        elif toward_delta < 0:
            toward_sign = -1.0

    for _ in range(num_steps):
        x += vx * dt
        y += vy * dt
        vy += gravity_y * dt

        if geom is not None and stop_on_reverse and prev_s is not None and toward_sign != 0.0:
            curr_s = geom.projection_s(x, y)
            ds = curr_s - prev_s
            # Discard the "coming back" tail once motion along wicket axis reverses.
            if ds * toward_sign < -min_projection_step:
                break
            prev_s = curr_s

        pts.append((x, y))
    return pts


def build_anchors_from_ball_infos(
    ball_infos: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for info in ball_infos:
        if info is None or info.get("ghost"):
            continue
        pos = info.get("interpolated_position")
        if pos is None or len(pos) < 2:
            continue
        fi = info.get("frame_idx")
        if fi is None:
            continue
        anchors.append(
            {
                "frame_idx": int(fi),
                "interpolated_position": (float(pos[0]), float(pos[1])),
            }
        )
    anchors.sort(key=lambda a: a["frame_idx"])
    return anchors


def _ball_xy(info: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if info is None:
        return None
    pos = info.get("interpolated_position")
    if pos is not None and len(pos) >= 2:
        return (float(pos[0]), float(pos[1]))
    box = info.get("box")
    if box and len(box) >= 4:
        x, y, w, h = box[:4]
        return (float(x + w / 2.0), float(y + h / 2.0))
    return None


def _point_in_box(px: float, py: float, box: List[float], margin: float = 8.0) -> bool:
    x, y, w, h = box
    return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)


def _pads_union_boxes(det_pads: List[Dict[str, Any]]) -> List[List[float]]:
    boxes: List[List[float]] = []
    for p in det_pads or []:
        if "Foot" in p.get("label", ""):
            continue
        b = p.get("box")
        if b and len(b) == 4:
            boxes.append([float(x) for x in b])
    return boxes


def distance_point_to_bbox(px: float, py: float, bbox: List[float]) -> float:
    """
    Minimum Euclidean distance from a point to an axis-aligned rectangle.
    bbox is [x, y, w, h] (top-left, size). Returns 0 if the point lies inside.
    """
    if len(bbox) < 4:
        return float("inf")
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return float(math.hypot(px - cx, py - cy))


def _impact_distance_to_targets(
    bx: float,
    by: float,
    pad_boxes: List[List[float]],
    batsman_box: Optional[List[int]],
    pad_preference_px: float = 5.0,
) -> float:
    """Smaller is closer; slight bias so pad is preferred when distances are similar."""
    d_pads = min((distance_point_to_bbox(bx, by, b) for b in pad_boxes), default=float("inf"))
    d_bat = float("inf")
    if batsman_box and len(batsman_box) == 4:
        d_bat = distance_point_to_bbox(bx, by, [float(v) for v in batsman_box])
    return min(d_pads, d_bat + pad_preference_px)


def _find_pad_body_impact_bbox_fallback(
    ball_infos: List[Optional[Dict[str, Any]]],
    pads_per_frame: List[List[Dict[str, Any]]],
    batsman_boxes: List[Optional[List[int]]],
    margin: float = 10.0,
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    """Legacy: first frame where detection center lies inside pad or batsman box."""
    n = min(len(ball_infos), len(pads_per_frame), len(batsman_boxes))
    for i in range(n):
        xy = _ball_xy(ball_infos[i])
        if xy is None:
            continue
        px, py = xy
        for b in _pads_union_boxes(pads_per_frame[i]):
            if _point_in_box(px, py, b, margin=margin):
                return i, xy
        bb = batsman_boxes[i]
        if bb and len(bb) == 4 and _point_in_box(px, py, [float(x) for x in bb], margin=margin):
            return i, xy
    return None, None


def _candidate_impact_frame_range(
    trajectory_model: TrajectoryModel,
    batsman_boxes: List[Optional[List[int]]],
    t_anchor_min: int,
    t_anchor_max: int,
    n_frames: int,
    near_region_px: float = 140.0,
    pre_contact_frames: int = 12,
) -> Tuple[int, int]:
    """
    Restrict search to frames from shortly before the ball enters the batsman
    region through the end of the anchor track.
    """
    t_lo = max(0, int(t_anchor_min))
    t_hi = min(n_frames - 1, int(t_anchor_max))
    if t_hi < t_lo:
        return t_lo, t_hi

    t_first_near: Optional[int] = None
    for t in range(t_lo, t_hi + 1):
        bb = batsman_boxes[t] if t < len(batsman_boxes) else None
        if not bb or len(bb) != 4:
            continue
        bp = predict_position(trajectory_model, float(t))
        if bp is None:
            continue
        if distance_point_to_bbox(bp[0], bp[1], [float(v) for v in bb]) <= near_region_px:
            t_first_near = t
            break

    if t_first_near is not None:
        t_lo = max(t_lo, t_first_near - pre_contact_frames)

    return t_lo, t_hi


def _refine_impact_subframe(
    trajectory_model: TrajectoryModel,
    t_center: float,
    pads_per_frame: List[List[Dict[str, Any]]],
    batsman_boxes: List[Optional[List[int]]],
    n_frames: int,
    pad_preference_px: float,
    step: float = 0.25,
) -> Tuple[float, Tuple[float, float]]:
    """Search [t_center-1, t_center+1] on the fitted curve for minimum distance."""
    t_lo = max(0.0, t_center - 1.0)
    t_hi = t_center + 1.0
    best_t = float(t_center)
    best_pt: Tuple[float, float] = (0.0, 0.0)
    best_d = float("inf")
    t = t_lo
    while t <= t_hi + 1e-9:
        bp = predict_position(trajectory_model, t)
        if bp is None:
            t += step
            continue
        fi = int(np.clip(int(round(t)), 0, n_frames - 1))
        pads = _pads_union_boxes(pads_per_frame[fi] if fi < len(pads_per_frame) else [])
        bbat = batsman_boxes[fi] if fi < len(batsman_boxes) else None
        d = _impact_distance_to_targets(bp[0], bp[1], pads, bbat, pad_preference_px)
        if d < best_d:
            best_d = d
            best_t = float(t)
            best_pt = (float(bp[0]), float(bp[1]))
        t += step
    return best_t, best_pt


def _snap_impact_point_to_trajectory(
    trajectory_model: TrajectoryModel,
    impact_frame: Optional[int],
    impact_point: Optional[Tuple[float, float]],
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    """
    If we fell back to a raw detection centre, replace the point with trajectory(t)
    so the overlay lies on the same curve as the fitted polyline.
    """
    if impact_frame is None or impact_point is None:
        return impact_frame, impact_point
    if not trajectory_model.segments:
        return impact_frame, impact_point
    p = predict_position(trajectory_model, float(impact_frame))
    if p is None:
        return impact_frame, impact_point
    return impact_frame, (float(p[0]), float(p[1]))


def find_pad_body_impact(
    ball_infos: List[Optional[Dict[str, Any]]],
    pads_per_frame: List[List[Dict[str, Any]]],
    batsman_boxes: List[Optional[List[int]]],
    trajectory_model: Optional[TrajectoryModel] = None,
    *,
    margin: float = 10.0,
    near_region_px: float = 140.0,
    pre_contact_frames: int = 12,
    max_reasonable_dist_px: float = 120.0,
    pad_preference_px: float = 5.0,
    subframe_step: float = 0.25,
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    """
    Impact frame/point via trajectory distance minimization when a fit exists;
    otherwise bbox first-contact fallback.

    Ball positions use predict_position(trajectory_model, t), not raw detection centers.
    """
    n_frames = min(len(ball_infos), len(pads_per_frame), len(batsman_boxes))
    if n_frames <= 0:
        return None, None

    if trajectory_model is None or not trajectory_model.segments:
        return _find_pad_body_impact_bbox_fallback(
            ball_infos, pads_per_frame, batsman_boxes, margin=margin
        )

    anchors = build_anchors_from_ball_infos(ball_infos)
    if not anchors:
        fi, pt = _find_pad_body_impact_bbox_fallback(
            ball_infos, pads_per_frame, batsman_boxes, margin=margin
        )
        return _snap_impact_point_to_trajectory(trajectory_model, fi, pt)

    t_anchor_min = anchors[0]["frame_idx"]
    t_anchor_max = anchors[-1]["frame_idx"]
    t_lo, t_hi = _candidate_impact_frame_range(
        trajectory_model,
        batsman_boxes,
        t_anchor_min,
        t_anchor_max,
        n_frames,
        near_region_px=near_region_px,
        pre_contact_frames=pre_contact_frames,
    )

    best_t_int: Optional[int] = None
    best_d = float("inf")

    for t in range(t_lo, t_hi + 1):
        bp = predict_position(trajectory_model, float(t))
        if bp is None:
            continue
        pads = _pads_union_boxes(pads_per_frame[t] if t < len(pads_per_frame) else [])
        bbat = batsman_boxes[t] if t < len(batsman_boxes) else None
        if not pads and (not bbat or len(bbat) != 4):
            continue
        d = _impact_distance_to_targets(bp[0], bp[1], pads, bbat, pad_preference_px)
        if d < best_d:
            best_d = d
            best_t_int = t

    if best_t_int is None or best_d > max_reasonable_dist_px:
        fi, pt = _find_pad_body_impact_bbox_fallback(
            ball_infos, pads_per_frame, batsman_boxes, margin=margin
        )
        return _snap_impact_point_to_trajectory(trajectory_model, fi, pt)

    t_ref, pt_ref = _refine_impact_subframe(
        trajectory_model,
        float(best_t_int),
        pads_per_frame,
        batsman_boxes,
        n_frames,
        pad_preference_px,
        step=subframe_step,
    )

    impact_frame = int(round(t_ref))
    impact_frame = int(np.clip(impact_frame, 0, n_frames - 1))
    return impact_frame, pt_ref


def sample_fitted_polyline(
    model: TrajectoryModel, t_min: int, t_max: int, step: int = 1
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for t in range(t_min, t_max + 1, step):
        p = predict_position(model, t)
        if p is not None:
            pts.append((float(p[0]), float(p[1])))
    return pts


def extrapolate_stump_intersection(
    model: TrajectoryModel,
    geom: WicketGeometry,
    t_start: float,
    max_steps: int = 800,
    dt: float = 0.35,
    gravity_y: float = _DEFAULT_GRAVITY_Y,
    stop_on_reverse: bool = True,
    min_projection_step: float = 1e-4,
) -> Tuple[Optional[Tuple[float, float]], bool, float]:
    """
    March forward with the same ballistic model as the purple preview line until
    the wicket-axis projection crosses the striker plane.
    """
    if not model.segments:
        return None, False, t_start

    seg = _segment_for_time(model, t_start)
    if seg is None:
        return None, False, t_start

    x = float(np.polyval(seg.x_coeffs, t_start))
    y = float(np.polyval(seg.y_coeffs, t_start))
    vx = float(2.0 * seg.x_coeffs[0] * t_start + seg.x_coeffs[1])
    vy = float(2.0 * seg.y_coeffs[0] * t_start + seg.y_coeffs[1])

    s_target = geom.s_stump
    prev_x, prev_y = x, y
    prev_s = geom.projection_s(x, y)
    toward_delta = s_target - prev_s
    if toward_delta > 0:
        toward_sign = 1.0
    elif toward_delta < 0:
        toward_sign = -1.0
    else:
        toward_sign = 0.0
    pseudo_t = float(t_start)

    for _ in range(max_steps):
        x += vx * dt
        y += vy * dt
        vy += gravity_y * dt
        pseudo_t += dt
        s = geom.projection_s(x, y)

        if stop_on_reverse and toward_sign != 0.0:
            ds = s - prev_s
            if ds * toward_sign < -min_projection_step:
                return None, False, pseudo_t

        if (prev_s - s_target) * (s - s_target) <= 0 and abs(s - prev_s) > 1e-6:
            alpha = abs(s_target - prev_s) / (abs(s - prev_s) + 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            ix = prev_x + alpha * (x - prev_x)
            iy = prev_y + alpha * (y - prev_y)
            hit_height = geom.stump_y_top - 8 <= iy <= geom.stump_y_bottom + 20
            lat = geom.inline_distance(ix, iy)
            hit_width = lat <= geom.lateral_threshold * 1.1
            return (ix, iy), bool(hit_height and hit_width), pseudo_t

        prev_x, prev_y, prev_s = x, y, s

    return None, False, pseudo_t


def analyze_lbw_sequence(
    ball_infos: List[Optional[Dict[str, Any]]],
    trajectory_model: TrajectoryModel,
    wickets_per_frame: List[List[Dict[str, Any]]],
    pads_per_frame: List[List[Dict[str, Any]]],
    batsman_boxes: List[Optional[List[int]]],
) -> Dict[str, Any]:
    """
    Produce LBW geometry verdicts and overlay primitives for visualization / API.
    When wickets are missing, still fills fitted_polyline + predicted_extension for debugging.
    """
    n_frames = len(ball_infos)
    out = _empty_overlay(n_frames)
    anchors = build_anchors_from_ball_infos(ball_infos)

    if not trajectory_model.segments:
        _mark_missing_data(out, "trajectory")
        return out

    bounce_f = trajectory_model.bounce_frame
    out["bounce_frame"] = bounce_f

    bounce_time = getattr(trajectory_model, "bounce_time", None)
    if bounce_time is not None:
        bounce_point = predict_position(trajectory_model, float(bounce_time))
        if bounce_point is not None:
            out["bounce_point"] = [float(bounce_point[0]), float(bounce_point[1])]
            out["bounce_time"] = float(bounce_time)

    pitch_point: Optional[Tuple[float, float]] = None
    pitch_frame_idx: Optional[int] = None
    if bounce_time is not None:
        pitch_point = predict_position(trajectory_model, float(bounce_time))
        pitch_frame_idx = int(bounce_f) if bounce_f is not None else int(np.floor(float(bounce_time)))
    elif bounce_f is not None:
        pitch_point = predict_position(trajectory_model, float(bounce_f))
        pitch_frame_idx = int(bounce_f)
    if pitch_point is None and trajectory_model.segments:
        t0 = trajectory_model.segments[0].start_frame
        pitch_point = predict_position(trajectory_model, float(t0))
        pitch_frame_idx = int(t0)
    if pitch_point is not None:
        out["pitch_point"] = [float(pitch_point[0]), float(pitch_point[1])]
        out["pitch_frame_idx"] = pitch_frame_idx

    impact_idx, impact_pt = find_pad_body_impact(
        ball_infos,
        pads_per_frame,
        batsman_boxes,
        trajectory_model,
    )
    if impact_pt is not None:
        out["impact_point"] = [float(impact_pt[0]), float(impact_pt[1])]
        out["impact_frame_idx"] = int(impact_idx) if impact_idx is not None else None

    t_min = 0
    t_max = max(0, n_frames - 1)
    if anchors:
        t_min = min(a["frame_idx"] for a in anchors)
        t_max = max(a["frame_idx"] for a in anchors)

    out["fitted_start_frame"] = int(t_min)
    fitted_end_frame = int(t_max)
    if impact_idx is not None:
        fitted_end_frame = max(int(t_min), min(int(t_max), int(impact_idx)))
    out["fitted_end_frame"] = int(fitted_end_frame)

    last_t = float(fitted_end_frame)

    fitted = sample_fitted_polyline(trajectory_model, int(t_min), int(fitted_end_frame), step=1)
    out["fitted_polyline"] = [[float(x), float(y)] for x, y in fitted]

    # Continue from last observed frame so the preview runs past the batsman toward stumps.
    ext_pts = sample_ballistic_extension(trajectory_model, t_start=last_t)
    out["predicted_extension"] = [[float(x), float(y)] for x, y in ext_pts]

    far_b, near_b = pick_reference_wicket_boxes(wickets_per_frame)
    if far_b is None or near_b is None:
        _mark_missing_data(out, "wicket pair")
        return out

    try:
        geom = wicket_geometry_from_boxes(far_b, near_b)
    except Exception:
        _mark_missing_data(out, "wicket geometry")
        return out

    line_seg = geom.extended_line_segment()
    out["wicket_line"] = [list(line_seg[0]), list(line_seg[1])]

    # Rebuild extension with wicket-axis direction gating so rebound segments are discarded.
    ext_pts = sample_ballistic_extension(
        trajectory_model,
        t_start=last_t,
        geom=geom,
        stop_on_reverse=True,
    )
    out["predicted_extension"] = [[float(x), float(y)] for x, y in ext_pts]

    pitch_inline = False
    if pitch_point is not None:
        pitch_inline = geom.inline_distance(pitch_point[0], pitch_point[1]) <= geom.lateral_threshold

    impact_inline = False
    if impact_pt is not None:
        impact_inline = geom.inline_distance(impact_pt[0], impact_pt[1]) <= geom.lateral_threshold

    stump_pt, wickets_hit, _t_cross = extrapolate_stump_intersection(
        trajectory_model,
        geom,
        t_start=last_t,
        stop_on_reverse=True,
    )
    if stump_pt is not None:
        out["stump_intersection"] = [float(stump_pt[0]), float(stump_pt[1])]

    out["pitch_inline"] = pitch_inline
    out["impact_inline"] = impact_inline
    out["wickets_hitting"] = wickets_hit

    geometric_out = pitch_inline and impact_inline and wickets_hit
    out["decision"] = "OUT" if geometric_out else "NOT OUT"
    out["geometric_lbw"] = geometric_out
    out["reason"] = None

    return out


def _empty_overlay(n_frames: int) -> Dict[str, Any]:
    return {
        "pitch_inline": False,
        "impact_inline": False,
        "wickets_hitting": False,
        "pitch_point": None,
        "bounce_point": None,
        "bounce_time": None,
        "pitch_frame_idx": None,
        "impact_point": None,
        "stump_intersection": None,
        "impact_frame_idx": None,
        "bounce_frame": None,
        "fitted_start_frame": 0,
        "fitted_end_frame": 0,
        "decision": NO_DECISION,
        "geometric_lbw": False,
        "fitted_polyline": [],
        "predicted_extension": [],
        "wicket_line": None,
        "reason": None,
    }


def lbw_overlay_for_api(overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Subset / string labels for PredictionService and JSON."""
    return {
        "pitch": "In-line" if overlay.get("pitch_inline") else "Outside",
        "impact": "In-line" if overlay.get("impact_inline") else "Outside",
        "wickets": "Hitting" if overlay.get("wickets_hitting") else "Missing",
        "decision": overlay.get("decision", NO_DECISION),
        "reason": overlay.get("reason"),
        "impact_frame_idx": overlay.get("impact_frame_idx"),
        "bounce_frame": overlay.get("bounce_frame"),
        "bounce_time": overlay.get("bounce_time"),
        "pitch_frame_idx": overlay.get("pitch_frame_idx"),
        "geometric_lbw": overlay.get("geometric_lbw", False),
    }

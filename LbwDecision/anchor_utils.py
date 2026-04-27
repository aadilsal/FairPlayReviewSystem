from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_anchors_from_ball_infos(ball_infos: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for ball_info in ball_infos:
        if ball_info is None:
            continue
        frame_idx = ball_info.get("frame_idx")
        interp_pos = ball_info.get("interpolated_position")
        if frame_idx is None or interp_pos is None:
            continue
        if not isinstance(interp_pos, (list, tuple)) or len(interp_pos) < 2:
            continue
        anchors.append(
            {
                "frame_idx": int(frame_idx),
                "interpolated_position": (float(interp_pos[0]), float(interp_pos[1])),
            }
        )
    return sorted(anchors, key=lambda a: a["frame_idx"])


def index_anchors_by_frame(anchors: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(a["frame_idx"]): a for a in anchors}


def ball_center_from_anchor(anchor: Dict[str, Any]) -> tuple[float, float]:
    x, y = anchor["interpolated_position"]
    return float(x), float(y)


def slice_anchors_before_frame(
    anchors: List[Dict[str, Any]], frame_idx: int, min_points: int = 3
) -> List[Dict[str, Any]]:
    subset = [a for a in anchors if int(a["frame_idx"]) <= int(frame_idx)]
    if len(subset) >= min_points:
        return subset
    return anchors[:]

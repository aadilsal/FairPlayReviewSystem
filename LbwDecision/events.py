from __future__ import annotations

from typing import Any, Dict, List, Optional


def point_inside_box(point: tuple[float, float], box: List[Any], margin: float = 0.0) -> bool:
    if not isinstance(box, list) or len(box) < 4:
        return False
    x, y, w, h = [float(v) for v in box[:4]]
    px, py = point
    return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)


def find_first_bounce_event(
    anchors: List[Dict[str, Any]],
    bounce_hint_frame: Optional[int] = None,
    min_v_down: float = 1.0,
    min_v_up: float = 1.0,
) -> Optional[Dict[str, Any]]:
    if len(anchors) < 3:
        if bounce_hint_frame is None:
            return None
        for a in anchors:
            if int(a["frame_idx"]) >= int(bounce_hint_frame):
                return {"frame_idx": int(a["frame_idx"]), "point": a["interpolated_position"]}
        return None

    sorted_anchors = sorted(anchors, key=lambda a: int(a["frame_idx"]))

    for i in range(len(sorted_anchors) - 2):
        a0 = sorted_anchors[i]
        a1 = sorted_anchors[i + 1]
        a2 = sorted_anchors[i + 2]

        dt1 = float(a1["frame_idx"] - a0["frame_idx"])
        dt2 = float(a2["frame_idx"] - a1["frame_idx"])
        if dt1 <= 0.0 or dt2 <= 0.0:
            continue

        y0 = float(a0["interpolated_position"][1])
        y1 = float(a1["interpolated_position"][1])
        y2 = float(a2["interpolated_position"][1])
        v1 = (y1 - y0) / dt1
        v2 = (y2 - y1) / dt2

        # Image coordinates: positive dy means moving downward.
        if v1 >= min_v_down and v2 <= -min_v_up:
            return {"frame_idx": int(a1["frame_idx"]), "point": a1["interpolated_position"]}

    if bounce_hint_frame is not None:
        for a in sorted_anchors:
            if int(a["frame_idx"]) >= int(bounce_hint_frame):
                return {"frame_idx": int(a["frame_idx"]), "point": a["interpolated_position"]}
    return None


def find_first_pad_impact_after_pitch(
    anchors_by_frame: Dict[int, Dict[str, Any]],
    pads_per_frame: List[List[Dict[str, Any]]],
    pitch_frame_idx: int,
    overlap_margin: float = 0.0,
) -> Optional[Dict[str, Any]]:
    contact = find_first_contact_after_pitch(
        anchors_by_frame,
        pads_per_frame,
        bats_per_frame=None,
        pitch_frame_idx=pitch_frame_idx,
        overlap_margin=overlap_margin,
    )
    if contact is None:
        return None
    if contact.get("target") != "pad":
        return None
    return {"frame_idx": contact["frame_idx"], "point": contact["point"]}


def find_first_contact_after_pitch(
    anchors_by_frame: Dict[int, Dict[str, Any]],
    pads_per_frame: List[List[Dict[str, Any]]],
    bats_per_frame: Optional[List[List[Dict[str, Any]]]],
    pitch_frame_idx: int,
    overlap_margin: float = 0.0,
) -> Optional[Dict[str, Any]]:
    for frame_idx in sorted(anchors_by_frame.keys()):
        if int(frame_idx) <= int(pitch_frame_idx):
            continue
        if frame_idx < 0 or frame_idx >= len(pads_per_frame):
            continue

        anchor = anchors_by_frame[frame_idx]
        point = anchor["interpolated_position"]

        # Bat-first precedence: if both overlap at same frame, treat as bat contact.
        frame_bats = []
        if bats_per_frame and frame_idx < len(bats_per_frame):
            frame_bats = bats_per_frame[frame_idx] or []
        for bat_det in frame_bats:
            box = bat_det.get("box", [])
            if point_inside_box(point, box, margin=overlap_margin):
                return {
                    "frame_idx": int(frame_idx),
                    "point": (float(point[0]), float(point[1])),
                    "target": "bat",
                }

        frame_pads = pads_per_frame[frame_idx] if pads_per_frame else []
        for pad_det in frame_pads:
            box = pad_det.get("box", [])
            if point_inside_box(point, box, margin=overlap_margin):
                return {
                    "frame_idx": int(frame_idx),
                    "point": (float(point[0]), float(point[1])),
                    "target": "pad",
                }
    return None

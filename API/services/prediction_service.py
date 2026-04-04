from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FRAME_FILE_RE = re.compile(r"frame_(\d+)\.json$", re.IGNORECASE)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ensure_project_import_paths() -> Path:
    root = Path(__file__).resolve().parents[2]
    for p in (root, root / "utils", root / "BallDetection"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def _point_in_box(px: float, py: float, box: List[float], margin: float = 10.0) -> bool:
    x, y, w, h = box
    return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)


class TrajectoryPostProcessor:
    """
    Backward-compatible lightweight trajectory smoother.
    Fills missing ball positions with linear interpolation/extrapolation.
    """

    @staticmethod
    def process_trajectory(track: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not track:
            return []

        corrected = [dict(item) for item in track]
        known_indices = [i for i, item in enumerate(corrected) if item.get("position") is not None]
        if not known_indices:
            return corrected

        first_known = known_indices[0]
        last_known = known_indices[-1]

        for i in range(0, first_known):
            corrected[i]["position"] = corrected[first_known]["position"]
            corrected[i]["source"] = "interpolated_leading"

        for i in range(last_known + 1, len(corrected)):
            corrected[i]["position"] = corrected[last_known]["position"]
            corrected[i]["source"] = "interpolated_trailing"

        for left, right in zip(known_indices, known_indices[1:]):
            gap = right - left
            if gap <= 1:
                continue

            lx, ly = corrected[left]["position"]
            rx, ry = corrected[right]["position"]
            for j in range(1, gap):
                t = j / float(gap)
                x = (1.0 - t) * float(lx) + t * float(rx)
                y = (1.0 - t) * float(ly) + t * float(ry)
                idx = left + j
                corrected[idx]["position"] = (x, y)
                corrected[idx]["source"] = "interpolated_linear"

        return corrected


class PredictionService:
    """
    Builds trajectory and geometric LBW analysis from per-frame metadata (same logic
    as offline detection_pipeline), plus legacy fields for the frontend.
    """

    @staticmethod
    def _load_frames_metadata(frames_dir: Path) -> List[Dict[str, Any]]:
        if not frames_dir.exists():
            raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

        frame_files: List[Tuple[int, Path]] = []
        for p in frames_dir.iterdir():
            if not p.is_file():
                continue
            m = _FRAME_FILE_RE.match(p.name)
            if not m:
                continue
            frame_files.append((int(m.group(1)), p))

        frame_files.sort(key=lambda x: x[0])

        frames: List[Dict[str, Any]] = []
        for _, fp in frame_files:
            with open(fp, "r", encoding="utf-8") as f:
                frames.append(json.load(f))
        return frames

    @staticmethod
    def _extract_ball_track(frame_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        track: List[Dict[str, Any]] = []
        for meta in frame_meta:
            frame_idx = int(meta.get("frame_index", 0))
            position: Optional[Tuple[float, float]] = None
            conf: float = 0.0
            source: str = "missing"

            for d in meta.get("detections", []):
                if d.get("label") != "Ball":
                    continue
                data = d.get("data") or {}
                ip = data.get("interpolated_position")
                if ip is not None and len(ip) >= 2:
                    position = (float(ip[0]), float(ip[1]))
                else:
                    box = data.get("box")
                    if box and len(box) == 4:
                        x, y, w, h = box
                        position = (float(x + w / 2.0), float(y + h / 2.0))
                conf = float(data.get("conf", conf))
                source = str(data.get("source") or source)
                break

            track.append(
                {
                    "frame_idx": frame_idx,
                    "position": position,
                    "conf": conf,
                    "source": source,
                }
            )
        return track

    @staticmethod
    def _extract_wickets(frame_meta: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        wickets_per_frame: List[List[Dict[str, Any]]] = []
        for meta in frame_meta:
            wicket_dets: List[Dict[str, Any]] = []
            for d in meta.get("detections", []):
                label = d.get("label", "")
                if isinstance(label, str) and "Wicket" in label and "box" in d:
                    wicket_dets.append(d)
            wickets_per_frame.append(wicket_dets)
        return wickets_per_frame

    @staticmethod
    def _extract_pads_per_frame(frame_meta: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for meta in frame_meta:
            pads: List[Dict[str, Any]] = []
            for d in meta.get("detections", []):
                if d.get("label") == "Pad" and "box" in d:
                    pads.append(d)
            out.append(pads)
        return out

    @staticmethod
    def _extract_batsman_boxes(frame_meta: List[Dict[str, Any]]) -> List[Optional[List[int]]]:
        boxes: List[Optional[List[int]]] = []
        for meta in frame_meta:
            bb: Optional[List[int]] = None
            for d in meta.get("detections", []):
                if d.get("label") == "Batsman" and "box" in d:
                    bb = [int(x) for x in d["box"]]
                    break
            boxes.append(bb)
        return boxes

    @staticmethod
    def _ball_infos_from_metadata(frame_meta: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        infos: List[Optional[Dict[str, Any]]] = []
        for meta in frame_meta:
            fi = int(meta.get("frame_index", 0))
            ball: Optional[Dict[str, Any]] = None
            for d in meta.get("detections", []):
                if d.get("label") != "Ball":
                    continue
                data = dict(d.get("data") or {})
                data["frame_idx"] = int(data.get("frame_idx", fi))
                ip = data.get("interpolated_position")
                if isinstance(ip, list) and len(ip) >= 2:
                    data["interpolated_position"] = (float(ip[0]), float(ip[1]))
                ball = data
                break
            infos.append(ball)
        return infos

    @staticmethod
    def predict_from_frames(frames_dir: Path) -> Dict[str, Any]:
        """
        Returns:
          impact, pitch, wickets, decision, confidence, impact_frame_idx,
          lbw (geometric summary), lbw_overlay (optional visualization payload).
        """
        _ensure_project_import_paths()
        from BallDetection.pipeline.trajectory import fit_trajectory
        from lbw_analyzer import (
            analyze_lbw_sequence,
            build_anchors_from_ball_infos,
            lbw_overlay_for_api,
        )

        frame_meta = PredictionService._load_frames_metadata(frames_dir)
        if not frame_meta:
            raise RuntimeError(f"No frame metadata JSON found in {frames_dir}")

        frame_meta = sorted(frame_meta, key=lambda m: int(m.get("frame_index", 0)))

        ball_infos = PredictionService._ball_infos_from_metadata(frame_meta)
        wickets_per_frame = PredictionService._extract_wickets(frame_meta)
        pads_per_frame = PredictionService._extract_pads_per_frame(frame_meta)
        batsman_boxes = PredictionService._extract_batsman_boxes(frame_meta)

        anchors = build_anchors_from_ball_infos(ball_infos)
        trajectory_model = fit_trajectory(anchors)
        lbw_overlay = analyze_lbw_sequence(
            ball_infos,
            trajectory_model,
            wickets_per_frame,
            pads_per_frame,
            batsman_boxes,
        )
        api_lbw = lbw_overlay_for_api(lbw_overlay)

        original_track = PredictionService._extract_ball_track(frame_meta)
        post = TrajectoryPostProcessor()
        corrected_track = post.process_trajectory(original_track)

        interpolated = sum(
            1 for r in corrected_track if str(r.get("source") or "").startswith("interpolated_")
        )
        total = max(1, len(corrected_track))
        reliability = _clamp01(1.0 - (interpolated / float(total)))

        geometric = bool(lbw_overlay.get("geometric_lbw"))
        missing = bool(lbw_overlay.get("reason"))
        if missing:
            base = 0.15
        else:
            base = 0.4 if geometric else 0.28
        confidence = _clamp01(base + 0.6 * reliability)

        impact_frame_idx = lbw_overlay.get("impact_frame_idx")

        return {
            "impact": api_lbw["impact"],
            "pitch": api_lbw["pitch"],
            "wickets": api_lbw["wickets"],
            "decision": api_lbw["decision"],
            "reason": api_lbw.get("reason"),
            "confidence": confidence,
            "impact_frame_idx": impact_frame_idx,
            "hit": bool(lbw_overlay.get("wickets_hitting")),
            "lbw": api_lbw,
            "geometric_lbw": geometric,
            "lbw_overlay": {
                k: lbw_overlay[k]
                for k in (
                    "pitch_inline",
                    "impact_inline",
                    "wickets_hitting",
                    "pitch_point",
                    "impact_point",
                    "stump_intersection",
                    "bounce_frame",
                    "decision",
                    "geometric_lbw",
                    "fitted_polyline",
                    "predicted_extension",
                    "wicket_line",
                    "reason",
                )
                if k in lbw_overlay
            },
        }

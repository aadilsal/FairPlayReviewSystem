from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FRAME_FILE_RE = re.compile(r"frame_(\d+)\.json$", re.IGNORECASE)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _point_in_box(px: float, py: float, box: List[float], margin: float = 10.0) -> bool:
    x, y, w, h = box
    return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)


class TrajectoryPostProcessor:
    """
    Backward-compatible lightweight trajectory smoother used by API prediction.
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

        # Fill leading missing positions with first known point.
        for i in range(0, first_known):
            corrected[i]["position"] = corrected[first_known]["position"]
            corrected[i]["source"] = "interpolated_leading"

        # Fill trailing missing positions with last known point.
        for i in range(last_known + 1, len(corrected)):
            corrected[i]["position"] = corrected[last_known]["position"]
            corrected[i]["source"] = "interpolated_trailing"

        # Fill gaps between known detections linearly.
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
    Builds a corrected ball trajectory from per-frame metadata, then derives
    the DRS-compatible fields expected by the frontend:
    impact, pitch, wickets, decision, confidence.
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
        # Input format expected by TrajectoryPostProcessor:
        # { frame_idx, position, conf, source }
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
    def predict_from_frames(frames_dir: Path) -> Dict[str, Any]:
        """
        Returns:
          {impact, pitch, wickets, decision, confidence, impact_frame_idx}
        """
        frame_meta = PredictionService._load_frames_metadata(frames_dir)
        if not frame_meta:
            raise RuntimeError(f"No frame metadata JSON found in {frames_dir}")

        original_track = PredictionService._extract_ball_track(frame_meta)
        wickets_per_frame = PredictionService._extract_wickets(frame_meta)

        post = TrajectoryPostProcessor()
        corrected_track = post.process_trajectory(original_track)

        # Determine first impact by intersection with any wicket bbox.
        hit = False
        impact_frame_idx: Optional[int] = None

        impact_ball_point: Optional[Tuple[float, float]] = None
        impact_wicket_box: Optional[List[float]] = None

        for i, corrected in enumerate(corrected_track):
            pos = corrected.get("position")
            if pos is None:
                continue
            bx, by = float(pos[0]), float(pos[1])

            for w in wickets_per_frame[i] if i < len(wickets_per_frame) else []:
                box = w.get("box")
                if not box or not isinstance(box, list) or len(box) != 4:
                    continue
                if _point_in_box(bx, by, box, margin=12.0):
                    hit = True
                    impact_frame_idx = int(corrected.get("frame_idx", i))
                    impact_ball_point = (bx, by)
                    impact_wicket_box = [float(x) for x in box]
                    break
            if hit:
                break

        wickets = "Hitting" if hit else "Missing"
        decision = "OUT" if hit else "NOT OUT"

        # In-line / Outside heuristics (2D): compare impact X with wicket center X.
        impact = "Outside"
        pitch = "Outside"
        if impact_ball_point and impact_wicket_box:
            bx, _ = impact_ball_point
            wx, wy, ww, wh = impact_wicket_box
            wicket_center_x = wx + ww / 2.0
            threshold = max(12.0, ww * 0.2)
            is_in_line = abs(bx - wicket_center_x) <= threshold
            impact = "In-line" if is_in_line else "Outside"
            pitch = "In-line" if is_in_line else "Outside"

        # Confidence: higher if less interpolation happened.
        total = max(1, len(corrected_track))
        interpolated = 0
        for r in corrected_track:
            src = str(r.get("source") or "")
            if src.startswith("interpolated_"):
                interpolated += 1

        reliability = _clamp01(1.0 - (interpolated / float(total)))
        # If we did hit, bump a bit; if missing, slightly lower.
        base = 0.35 if hit else 0.25
        confidence = _clamp01(base + 0.65 * reliability)

        return {
            "impact": impact,
            "pitch": pitch,
            "wickets": wickets,
            "decision": decision,
            "confidence": confidence,
            "impact_frame_idx": impact_frame_idx,
            "hit": hit,
        }


"""
detector.py
------------------
YOLO interface and anchor selection for WicketTracker.
Owns everything that touches the model or decides which detection is the
near (anchor) wicket.  No far-wicket logic lives here.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from typing import Optional

from geometry import box_center_y, smooth


class WicketDetector:
    def __init__(
        self,
        model_path: str = "weights/wicket_weights.pt",
        min_det_conf: float = 0.25,
        smooth_top_alpha: float = 0.2,
        max_anchor_miss: int = 60,
    ):
        self.model = YOLO(model_path)
        self.min_det_conf = min_det_conf
        self.smooth_top_alpha = smooth_top_alpha
        self.max_anchor_miss = max_anchor_miss

        # ── Anchor state ──────────────────────────────────────────────────────
        self.anchor_top_smoothed: Optional[np.ndarray] = None
        self.last_anchor_det: Optional[dict] = None
        self.anchor_miss_count: int = 0

    # ── YOLO ──────────────────────────────────────────────────────────────────

    def run(self, frame: np.ndarray, conf: float) -> list[dict]:
        """Run YOLO on `frame` and return a flat list of detection dicts."""
        results = self.model.predict(frame, conf=conf, verbose=False)
        dets = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                c = float(box.conf[0])
                dets.append({
                    "box": [x1, y1, w, h],
                    "conf": c,
                    "area": w * h,
                })

        return dets

    # ── Anchor selection ──────────────────────────────────────────────────────

    def select_anchor(
        self, dets: list[dict], h_img: int, frame: np.ndarray
    ) -> Optional[dict]:
        """
        Pick the largest detection whose vertical centre is in the lower half
        of the frame — that's the near wicket.

        Falls back to the last known anchor for up to `max_anchor_miss` frames,
        drawing it in grey to signal that it's stale.
        """
        candidates = [
            d for d in dets
            if box_center_y(d["box"]) > h_img * 0.45
        ]
        candidates.sort(key=lambda d: d["area"], reverse=True)

        if candidates:
            anchor = candidates[0]
            self.last_anchor_det = anchor
            self.anchor_miss_count = 0
            return anchor

        # ── Fallback: reuse last seen anchor ──────────────────────────────────
        if self.last_anchor_det is not None and self.anchor_miss_count < self.max_anchor_miss:
            self.anchor_miss_count += 1
            ax, ay, aw, ah = self.last_anchor_det["box"]
            cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (100, 100, 100), 2)
            return self.last_anchor_det

        return None

    # ── Anchor smoothing ──────────────────────────────────────────────────────

    def update_anchor_smoothing(self, anchor: dict) -> None:
        """
        Smooth the anchor's top-left position with an EMA.
        Skips the update when the raw position is within 3 px of the current
        smoothed value to avoid jitter from sub-pixel noise.
        """
        anchor_top_raw = np.array(
            [float(anchor["box"][0]), float(anchor["box"][1])], dtype=float
        )

        if self.anchor_top_smoothed is None:
            self.anchor_top_smoothed = anchor_top_raw
            return

        if np.linalg.norm(anchor_top_raw - self.anchor_top_smoothed) < 3.0:
            return

        self.anchor_top_smoothed = smooth(
            self.anchor_top_smoothed,
            anchor_top_raw,
            self.smooth_top_alpha,
        )

    # ── Far candidate ─────────────────────────────────────────────────────────

    def get_far_candidate(
        self, dets: list[dict], anchor: dict
    ) -> Optional[dict]:
        """
        From all detections, return the largest one that sits *above* the
        smoothed anchor top and doesn't heavily overlap the anchor (IoU < 0.45).
        That's our far-wicket candidate.
        """
        if self.anchor_top_smoothed is None:
            return None

        far_candidates = []
        ax, ay, aw, ah = anchor["box"]

        for d in dets:
            tx, ty = float(d["box"][0]), float(d["box"][1])

            if ty >= self.anchor_top_smoothed[1]:
                continue  # not above anchor

            bx, by, bw, bh = d["box"]
            xA = max(bx, ax)
            yA = max(by, ay)
            xB = min(bx + bw, ax + aw)
            yB = min(by + bh, ay + ah)

            inter = max(0, xB - xA) * max(0, yB - yA)
            iou = inter / (bw * bh + aw * ah - inter + 1e-6)

            if iou < 0.45:
                far_candidates.append(d)

        far_candidates.sort(key=lambda d: d["area"], reverse=True)
        return far_candidates[0] if far_candidates else None

    # ── State reset ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.anchor_top_smoothed = None
        self.last_anchor_det = None
        self.anchor_miss_count = 0
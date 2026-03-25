"""
wicket_tracker.py
-----------------
Thin orchestrator — wires together detector, far handler, and drawing.
No geometry math, no cv2 calls, no raw state manipulation lives here.
"""

import numpy as np
from typing import Optional

from state import WicketTrackerState
from detector import WicketDetector
from far_handler import FarWicketHandler
from drawer import draw_anchor
from geometry import compute_iou
from drawer import draw_anchor, draw_far

class WicketTrackerV5:
    def __init__(
        self,
        model_path: str = "weights/wicket_weights.pt",
        min_det_conf: float = 0.25,
        # smoothing / clamp overrides
        smooth_top_alpha: float = 0.2,
        width_alpha: float = 0.3,
        width_px_cap: int = 2,
        initial_width_ratio: float = 0.30,
        max_width_expand_factor: float = 500,
        max_dx_per_frame: float = 10,
        max_dy_per_frame: float = 2,
        max_anchor_miss: int = 60,
        min_det_height_ratio_full: float = 0.75,
    ):
        # ── Shared state ──────────────────────────────────────────────────────
        self.state = WicketTrackerState(
            smooth_top_alpha=smooth_top_alpha,
            width_alpha=width_alpha,
            width_px_cap=width_px_cap,
            initial_width_ratio=initial_width_ratio,
            max_width_expand_factor=max_width_expand_factor,
            max_dx=max_dx_per_frame,
            max_dy=max_dy_per_frame,
            max_anchor_miss=max_anchor_miss,
            min_det_height_ratio_full=min_det_height_ratio_full,
            min_det_conf=min_det_conf,
        )

        # ── Sub-systems ───────────────────────────────────────────────────────
        self.detector = WicketDetector(
            model_path=model_path,
            min_det_conf=min_det_conf,
            smooth_top_alpha=smooth_top_alpha,
            max_anchor_miss=max_anchor_miss,
        )
        self.far_handler = FarWicketHandler(self.state)

    # ── Main entry point ──────────────────────────────────────────────────────

    def detect_and_track(
        self, frame: np.ndarray, conf: float = 0.25
    ) -> tuple[np.ndarray, list]:
        h_img, w_img = frame.shape[:2]

        # 1. Run YOLO
        dets = self.detector.run(frame, conf)
        if not dets:
            return frame, []

        # 2. Pick anchor (near wicket)
        anchor = self.detector.select_anchor(dets, h_img, frame)
        if anchor is None:
            return frame, []

        # 3. Smooth anchor position
        self.detector.update_anchor_smoothing(anchor)

        # 4. Draw anchor
        # draw_anchor(frame, anchor["box"])

        # 5. Find far candidate
        far = self.detector.get_far_candidate(dets, anchor)

        # 6. Seed output with near wicket
        final_dets = [{"label": "Wicket_Near", "box": anchor["box"], "conf": anchor["conf"]}]

        # 7. Far wicket — three paths
        anchor_top_smoothed = self.detector.anchor_top_smoothed

        if self.far_handler.is_far_full(far):
            frame, final_dets = self.far_handler.handle_far_full(
                frame, far, anchor, anchor_top_smoothed, final_dets, w_img
            )
        elif not self.state.locked:
            frame, final_dets = self.far_handler.handle_not_locked(
                frame, far, anchor, final_dets, w_img
            )
        else:
            frame, final_dets = self.far_handler.handle_prediction(
                frame, anchor, anchor_top_smoothed, final_dets, w_img, h_img
            )

        # 8. Merge if near and far overlap the same wicket
        final_dets = self._merge_if_same_wicket(final_dets)

        # 9. Draw visualisations at the end

        for det in final_dets:
            if det["label"] == "Wicket_Near":
                draw_anchor(frame, det["box"])
            elif det["label"] == "Wicket_Far":
                mode = det.get("mode", "")
                color = det.get("color", (0, 255, 255))
                draw_far(frame, det["box"], mode, color)
        
        return frame, final_dets

    # ── Merge ─────────────────────────────────────────────────────────────────

    def _merge_if_same_wicket(self, final_dets: list) -> list:
        """
        If near and far boxes overlap heavily (IoU > 0.2) they are almost
        certainly the same physical wicket — collapse them into one.
        """
        near = next((d for d in final_dets if d["label"] == "Wicket_Near"), None)
        far  = next((d for d in final_dets if d["label"] == "Wicket_Far"),  None)

        if near and far and compute_iou(near["box"], far["box"]) > 0.2:
            far_merged = dict(far)
            far_merged["conf"] = max(near["conf"], far["conf"])
            return [far_merged]

        return final_dets

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Full reset — use between innings / camera cuts."""
        self.state.reset_all()
        self.detector.reset()


# ── Module-level singleton ────────────────────────────────────────────────────

_tracker: Optional[WicketTrackerV5] = None


def detect_wicket(
    frame: np.ndarray, conf: float = 0.25
) -> tuple[np.ndarray, list]:
    global _tracker
    if _tracker is None:
        _tracker = WicketTrackerV5(min_det_conf=conf)
    return _tracker.detect_and_track(frame, conf)
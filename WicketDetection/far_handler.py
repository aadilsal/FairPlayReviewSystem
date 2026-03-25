"""
wicket_far_handler.py
---------------------
Far-wicket lifecycle: lock initialisation, state updates, box building,
and the three execution paths (full detection / partial detection / prediction).

Depends on:
    geometry  — pure math helpers
    drawer   — cv2 rendering (imported lazily per-call to avoid
                       circular imports if drawing ever needs handler types)

State is passed in explicitly via WicketTrackerState so this class owns
no frame-to-frame memory of its own.
"""

import numpy as np
from typing import Optional

from geometry import (
    top_left,
    smooth,
    clamp_step,
    is_full_detection,
    aspect_ratio_ok,
)
from state import WicketTrackerState


class FarWicketHandler:
    def __init__(self, state: WicketTrackerState):
        self.s = state  # shared mutable state

    # ── Gate: is this detection a full far-wicket view? ───────────────────────

    def is_far_full(self, far: Optional[dict]) -> bool:
        """
        True when `far` looks like a complete, unoccluded far-wicket detection:
          - confidence above threshold
          - height passes the full-detection test against known height
          - aspect ratio is plausible for a standing wicket
        """
        if not far or far["conf"] < self.s.min_det_conf:
            return False

        _, _, fw, fh = far["box"]
        expected_h = int(self.s.prev_far_height) if self.s.prev_far_height is not None else None

        if not is_full_detection(fh, expected_h, self.s.min_det_height_ratio_full):
            return False

        if not aspect_ratio_ok(fw, fh):
            return False

        return True

    # ── Path 1: full detection ────────────────────────────────────────────────

    def handle_far_full(
        self,
        frame: np.ndarray,
        far: dict,
        anchor: dict,
        anchor_top_smoothed: np.ndarray,
        final_dets: list,
        w_img: int,
    ) -> tuple[np.ndarray, list]:
        """
        Happy path — detector gave us a confident, full-height far wicket.
        Initialise or update the lock, then build and emit the far box.
        """

        fx, fy, fw, fh = far["box"]
        far_top = top_left(far["box"])

        # If locked and the new detection has jumped too far, fall back to prediction
        if self.s.locked and self.s.last_known_top is not None:
            dist = np.linalg.norm(far_top - self.s.last_known_top)
            if dist > 20:
                return self.handle_prediction(frame, anchor, anchor_top_smoothed, final_dets, w_img, frame.shape[0])

        if not self.s.locked:
            self._initialize_lock(anchor, far, far_top, anchor_top_smoothed)
        else:
            self._update_locked_state(anchor, far, far_top, fw, fh)

        far_box = self._build_far_box(fx, w_img, far_top)
        # draw_far(frame, far_box, "(Det)", (0, 128, 255))
        final_dets.append({
            "label": "Wicket_Far",
            "box": far_box,
            "conf": 0.95,
            "mode": "(Det)",
            "color": (0, 128, 255)
        })

        return frame, final_dets

    # ── Path 2: partial / low-confidence detection (not yet locked) ───────────

    def handle_not_locked(
        self,
        frame: np.ndarray,
        far: Optional[dict],
        anchor: dict,
        final_dets: list,
        w_img: int,
    ) -> tuple[np.ndarray, list]:
        """
        We have a detection but it hasn't passed the full-detection gate yet,
        and we haven't locked a reference geometry.  Accept what we can measure
        but clamp width conservatively.
        """
        from drawer import draw_far

        if not far or far["conf"] < self.s.min_det_conf:
            return frame, final_dets

        fx, fy, fw, fh = far["box"]

        est_w = int(anchor["box"][2] * self.s.initial_width_ratio)
        adjusted_w = max(fw, est_w)
        adjusted_w = min(adjusted_w, int(fw * self.s.max_width_expand_factor))

        if self.s.prev_far_width is not None:
            adjusted_w = int(np.clip(
                adjusted_w,
                self.s.prev_far_width - self.s.width_px_cap,
                self.s.prev_far_width + self.s.width_px_cap,
            ))

        if fx + adjusted_w > w_img:
            adjusted_w = max(4, w_img - fx)

        far_box = [int(fx), int(fy), int(adjusted_w), int(fh)]

        self.s.prev_far_width = far_box[2]
        self.s.prev_far_height = far_box[3]
        self.s.last_known_top = top_left(far_box)

        # draw_far(frame, far_box, "(PartialDet)", (0, 128, 200))
        final_dets.append({
            "label": "Wicket_Far",
            "box": far_box,
            "conf": 0.8,
            "mode": "(PartialDet)",
            "color": (0, 128, 200)
        })

        return frame, final_dets

    # ── Path 3: no usable detection — predict from anchor motion ─────────────

    def handle_prediction(
        self,
        frame: np.ndarray,
        anchor: dict,
        anchor_top_smoothed: np.ndarray,
        final_dets: list,
        w_img: int,
        h_img: int,
    ) -> tuple[np.ndarray, list]:
        """
        Detector missed the far wicket (or the detection jumped implausibly).
        Predict its position by scaling the locked offset with how much the
        anchor has grown/shrunk since lock time.
        """
        from drawer import draw_far

        if self.s.ref_anchor_width is None:
            return frame, final_dets

        scale = float(anchor["box"][2]) / (self.s.ref_anchor_width + 1e-9)

        # ── Width ─────────────────────────────────────────────────────────────
        if self.s.prev_far_width is None:
            pred_w = int(self.s.ref_anchor_width * self.s.initial_width_ratio)
        else:
            pred_w = int(self.s.prev_far_width * (1.0 + (scale - 1.0) * 0.05))
            pred_w = int(np.clip(
                pred_w,
                self.s.prev_far_width - self.s.width_px_cap,
                self.s.prev_far_width + self.s.width_px_cap,
            ))

        # ── Height ────────────────────────────────────────────────────────────
        pred_h = int(self.s.prev_far_height) if self.s.prev_far_height is not None else int(pred_w * 3.11)

        # ── Position ──────────────────────────────────────────────────────────
        predicted_top = anchor_top_smoothed + self.s.ref_offset_top * scale

        anchor_for_clamp = (
            self.s.last_good_top
            if self.s.last_good_top is not None
            else (self.s.last_known_top if self.s.last_known_top is not None else predicted_top)
        )

        clamped_top = clamp_step(anchor_for_clamp, predicted_top, self.s.max_dx, self.s.max_dy)

        self.s.last_known_top = (
            clamped_top.copy()
            if self.s.last_known_top is None
            else smooth(self.s.last_known_top, clamped_top, 0.18)
        )

        self.s.prev_far_width = int(pred_w)
        self.s.prev_far_height = int(pred_h)

        # ── Clamp to frame bounds ─────────────────────────────────────────────
        fx = max(0, int(self.s.last_known_top[0]))
        fy = int(self.s.last_known_top[1])

        if fx + pred_w > w_img:
            pred_w = max(4, w_img - fx)

        bottom_limit = int(h_img * 0.98)
        if fy + pred_h > bottom_limit:
            pred_h = max(4, bottom_limit - fy)

        far_box = [fx, fy, int(pred_w), int(pred_h)]

        # draw_far(frame, far_box, "(Pred)", (0, 255, 255))
        final_dets.append({
            "label": "Wicket_Far",
            "box": far_box,
            "conf": 1.0,
            "mode": "(Pred)",
            "color": (0, 255, 255)
        })

        return frame, final_dets

    # ── Lock helpers ──────────────────────────────────────────────────────────

    def _initialize_lock(
        self,
        anchor: dict,
        far: dict,
        far_top: np.ndarray,
        anchor_top_smoothed: np.ndarray,
    ) -> None:
        """Record reference geometry on the first good far-wicket detection."""
        fx, fy, fw, fh = far["box"]

        self.s.locked = True
        self.s.ref_anchor_width = anchor["box"][2]
        self.s.ref_offset_top = far_top - anchor_top_smoothed

        self.s.prev_far_width = int(fw)
        self.s.prev_far_height = int(fh)
        self.s.last_good_top = far_top.copy()
        self.s.last_known_top = far_top.copy()
        self.s.prev_far_area = fw * fh

    def _update_locked_state(
        self,
        anchor: dict,
        far: dict,
        far_top: np.ndarray,
        fw: float,
        fh: float,
    ) -> None:
        """Blend new detection into smoothed state while we're locked."""
        # ── Position ──────────────────────────────────────────────────────────
        if self.s.last_good_top is None:
            self.s.last_good_top = far_top.copy()
        else:
            mismatch = np.linalg.norm(far_top - self.s.last_good_top)
            alpha = 0.28 if mismatch < 10 else 0.12
            self.s.last_good_top = smooth(self.s.last_good_top, far_top, alpha)

        # ── Width — only expand rightward, never shrink ───────────────────────
        if self.s.prev_far_width is None:
            est_w = int(anchor["box"][2] * self.s.initial_width_ratio)
            adjusted_w = max(fw, est_w)
        else:
            adjusted_w = max(fw, self.s.prev_far_width)

        adjusted_w = min(adjusted_w, int(fw * self.s.max_width_expand_factor))

        if self.s.prev_far_width is None:
            width_used = int(adjusted_w)
        else:
            width_used = int(
                self.s.prev_far_width * (1.0 - self.s.width_alpha)
                + adjusted_w * self.s.width_alpha
            )
            width_used = int(np.clip(
                width_used,
                self.s.prev_far_width - self.s.width_px_cap,
                self.s.prev_far_width + self.s.width_px_cap,
            ))

        # ── Height — never reduce from known height ───────────────────────────
        if self.s.prev_far_height is None:
            height_used = int(fh)
        elif fh > self.s.prev_far_height:
            height_used = int(self.s.prev_far_height * (1.0 - 0.12) + fh * 0.12)
        else:
            height_used = int(self.s.prev_far_height)

        self.s.prev_far_width = width_used
        self.s.prev_far_height = height_used
        self.s.prev_far_area = width_used * height_used

        # ── Blend last_known_top toward last_good_top ─────────────────────────
        self.s.last_known_top = (
            self.s.last_good_top.copy()
            if self.s.last_known_top is None
            else smooth(self.s.last_known_top, self.s.last_good_top, 0.25)
        )

    # ── Box builder ───────────────────────────────────────────────────────────

    def _build_far_box(
        self, left_x: float, w_img: int, fallback_top: np.ndarray
    ) -> list:
        """
        Assemble the final [x, y, w, h] far box using smoothed state,
        clamping width so the box never overflows the frame edge.
        """
        width_final = int(self.s.prev_far_width)
        height_final = int(self.s.prev_far_height)

        if left_x + width_final > w_img:
            width_final = max(4, w_img - int(left_x))

        top = self.s.last_known_top if self.s.last_known_top is not None else fallback_top

        return [int(left_x), int(top[1]), width_final, height_final]
"""
state.py
---------------
Pure state container for WicketTracker.
No logic, no cv2, no YOLO — just the data that persists across frames.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class WicketTrackerState:
    # ── Smoothing / clamp config ──────────────────────────────────────────────
    smooth_top_alpha: float = 0.2
    width_alpha: float = 0.3
    width_px_cap: int = 2                  # +/- pixels allowed per frame
    initial_width_ratio: float = 0.30      # far width ≈ anchor_width * ratio
    max_width_expand_factor: float = 500   # safety cap relative to detected width
    max_dx: float = 10
    max_dy: float = 2
    min_det_height_ratio_full: float = 0.75
    min_det_conf: float = 0.25

    # ── Anchor miss tracking ──────────────────────────────────────────────────
    max_anchor_miss: int = 60
    anchor_miss_count: int = 0
    last_anchor_det: Optional[dict] = field(default=None, repr=False)

    # ── Anchor smoothing ──────────────────────────────────────────────────────
    anchor_top_smoothed: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Lock state ────────────────────────────────────────────────────────────
    # Once the far wicket has been confidently seen, we lock its reference
    # geometry so we can predict its position when it becomes partially hidden.
    locked: bool = False
    ref_anchor_width: Optional[float] = None   # anchor width at lock time
    ref_offset_top: Optional[np.ndarray] = field(default=None, repr=False)  # far_top - anchor_top at lock time

    # ── Far wicket geometry memory ────────────────────────────────────────────
    last_good_top: Optional[np.ndarray] = field(default=None, repr=False)   # smoothed from full detections only
    last_known_top: Optional[np.ndarray] = field(default=None, repr=False)  # blended from good_top + predictions
    prev_far_width: Optional[int] = None
    prev_far_height: Optional[int] = None
    prev_far_area: Optional[float] = None

    # ── Convenience ───────────────────────────────────────────────────────────
    def reset_far(self) -> None:
        """Wipe all far-wicket memory (call if tracking is lost unrecoverably)."""
        self.locked = False
        self.ref_anchor_width = None
        self.ref_offset_top = None
        self.last_good_top = None
        self.last_known_top = None
        self.prev_far_width = None
        self.prev_far_height = None
        self.prev_far_area = None

    def reset_anchor(self) -> None:
        """Wipe anchor memory."""
        self.anchor_miss_count = 0
        self.last_anchor_det = None
        self.anchor_top_smoothed = None

    def reset_all(self) -> None:
        self.reset_far()
        self.reset_anchor()
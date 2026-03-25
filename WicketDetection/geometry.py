"""
wicket_geometry.py
------------------
Stateless math / geometry utilities for WicketTracker.
No cv2, no YOLO, no state — pure functions only.
"""

import numpy as np
from typing import Optional


# ── Box accessors ─────────────────────────────────────────────────────────────

def top_left(box: list) -> np.ndarray:
    """Return the top-left corner of an [x, y, w, h] box as a float array."""
    x, y, w, h = box
    return np.array([float(x), float(y)], dtype=float)


def top_center(box: list) -> np.ndarray:
    """Return the top-center point of an [x, y, w, h] box as a float array."""
    x, y, w, h = box
    return np.array([x + w / 2.0, float(y)], dtype=float)


def box_center_y(box: list) -> float:
    """Return the vertical center of an [x, y, w, h] box."""
    x, y, w, h = box
    return y + h / 2.0


# ── Smoothing / clamping ──────────────────────────────────────────────────────

def smooth(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average.
    alpha=1.0 → snap to new immediately; alpha→0 → barely moves.
    Returns a copy of `new` if `prev` is None (first call).
    """
    if prev is None:
        return new.copy()
    return prev * (1.0 - alpha) + new * alpha


def clamp_step(
    prev: Optional[np.ndarray],
    target: np.ndarray,
    max_dx: float,
    max_dy: float,
) -> np.ndarray:
    """
    Move from `prev` toward `target` but cap the per-frame displacement
    to (max_dx, max_dy).  Returns a copy of `target` if `prev` is None.
    """
    if prev is None:
        return target.copy()
    dx = np.clip(target[0] - prev[0], -max_dx, max_dx)
    dy = np.clip(target[1] - prev[1], -max_dy, max_dy)
    return np.array([prev[0] + dx, prev[1] + dy], dtype=float)


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou(box1: list, box2: list) -> float:
    """
    Intersection-over-Union for two [x, y, w, h] boxes.
    Returns a value in [0, 1].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    inter = max(0, xB - xA) * max(0, yB - yA)
    union = w1 * h1 + w2 * h2 - inter + 1e-6

    return inter / union


# ── Detection quality helpers ─────────────────────────────────────────────────

def is_full_detection(
    det_h: float,
    expected_h: Optional[float],
    min_ratio: float = 0.75,
) -> bool:
    """
    True if `det_h` looks like a complete (not partially-occluded) wicket.
    Falls back to a raw pixel threshold when no expected height is known yet.
    """
    if expected_h is None:
        return det_h > 12
    return det_h >= expected_h * min_ratio


def aspect_ratio_ok(w: float, h: float, lo: float = 1.0, hi: float = 6.0) -> bool:
    """True if h/w is within the plausible range for a standing wicket."""
    ar = h / (w + 1e-9) if w > 0 else 0.0
    return lo < ar < hi
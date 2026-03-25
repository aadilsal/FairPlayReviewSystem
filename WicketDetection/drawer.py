"""
drawer.py
-----------------
All cv2 rendering for WicketTracker.
No state, no YOLO, no geometry logic — pure draw calls only.
"""

import cv2
import numpy as np
from typing import Union


# ── Type alias ────────────────────────────────────────────────────────────────

Color = tuple[int, int, int]   # BGR


# ── Near wicket (anchor) ──────────────────────────────────────────────────────

def draw_anchor(frame: np.ndarray, box: list) -> None:
    """Draw the near-wicket (anchor) detection in green."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame, "Wicket_Near",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
    )


# ── Far wicket ────────────────────────────────────────────────────────────────

def draw_far(frame: np.ndarray, box: list, label_suffix: str, color: Color) -> None:
    """
    Draw the far-wicket box with a mode suffix in the label.

    label_suffix examples:
        "(Det)"         — live detection, full confidence
        "(PartialDet)"  — low-confidence / partial detection
        "(Pred)"        — pure prediction, no detection this frame
    """
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame, f"Wicket_Far {label_suffix}",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
    )


# ── Convenience wrappers (named by mode) ──────────────────────────────────────

def draw_far_detected(frame: np.ndarray, box: list) -> None:
    draw_far(frame, box, "(Det)", (0, 128, 255))


def draw_far_partial(frame: np.ndarray, box: list) -> None:
    draw_far(frame, box, "(PartialDet)", (0, 128, 200))


def draw_far_predicted(frame: np.ndarray, box: list) -> None:
    draw_far(frame, box, "(Pred)", (0, 255, 255))


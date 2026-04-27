"""
Code-drawn LBW review slate: scene + structured Pitch / Impact / Wickets / Decision panels.
Uses OpenCV only (no external template image).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _fit_image_to_box(
    img: np.ndarray, box_w: int, box_h: int, pad: int = 0
) -> Tuple[np.ndarray, int, int]:
    """Resize with aspect ratio; letterbox with black bars. Returns image, offset_x, offset_y."""
    h, w = img.shape[:2]
    inner_w = max(1, box_w - 2 * pad)
    inner_h = max(1, box_h - 2 * pad)
    scale = min(inner_w / float(w), inner_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    ox = (box_w - nw) // 2
    oy = (box_h - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out, ox, oy


def _panel_color_for_value(label: str, value: str) -> Tuple[int, int, int]:
    """BGR accent for card border / title bar."""
    v = (value or "").lower()
    if "in-line" in v or v == "hitting":
        return (80, 180, 80)
    if "outside" in v or v == "missing":
        return (60, 60, 220)
    return (180, 140, 60)


def _decision_colors(decision: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """(title_bar_bgr, border_bgr) for decision strip."""
    d = (decision or "").upper()
    if d == "OUT":
        return ((40, 120, 40), (60, 200, 60))
    if d == "NOT OUT":
        return ((40, 60, 160), (60, 80, 220))
    return ((50, 110, 140), (70, 150, 200))


def _fit_point_to_box(
    px: float,
    py: float,
    src_w: int,
    src_h: int,
    box_w: int,
    box_h: int,
    pad: int = 0,
) -> Tuple[int, int]:
    """Map a source-image point into a fitted letterboxed box."""
    inner_w = max(1, box_w - 2 * pad)
    inner_h = max(1, box_h - 2 * pad)
    scale = min(inner_w / float(src_w), inner_h / float(src_h))
    nw = max(1, int(round(src_w * scale)))
    nh = max(1, int(round(src_h * scale)))
    ox = (box_w - nw) // 2
    oy = (box_h - nh) // 2
    return int(round(ox + px * scale)), int(round(oy + py * scale))


def render_lbw_review_card(
    scene_bgr: np.ndarray,
    api: Dict[str, Any],
    overlay: Dict[str, Any],
    *,
    frame_index: Optional[int] = None,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
) -> np.ndarray:
    """
    Compose a 16:9 review card: left = fitted scene, right = four verdict panels.

    api: lbw_overlay_for_api() dict (pitch, impact, wickets, decision, reason, …)
    overlay: full lbw overlay (optional extras; uses geometric_lbw, reason)
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (36, 32, 28)  # dark warm gray BGR

    margin = 20
    header_h = 56
    footer_h = 44
    split_x = int(canvas_w * 0.60)
    scene_box_w = split_x - 2 * margin
    scene_box_h = canvas_h - header_h - footer_h - 2 * margin

    # Header strip
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (48, 44, 40), -1)
    title = "LBW Review — AI Analysis"
    cv2.putText(
        canvas,
        title,
        (margin, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        (240, 240, 245),
        2,
        cv2.LINE_AA,
    )
    if frame_index is not None:
        ft = f"Frame {int(frame_index)}"
        tw, _ = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            canvas,
            ft,
            (canvas_w - tw - margin, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 210),
            2,
            cv2.LINE_AA,
        )

    # Scene (left)
    scene_y0 = header_h + margin
    scene_fit, _, _ = _fit_image_to_box(scene_bgr, scene_box_w, scene_box_h, pad=4)
    canvas[scene_y0 : scene_y0 + scene_box_h, margin : margin + scene_box_w] = scene_fit

    wicket_line = overlay.get("wicket_line") if overlay else None
    if wicket_line and len(wicket_line) == 2:
        src_h, src_w = scene_bgr.shape[:2]
        p0 = _fit_point_to_box(
            float(wicket_line[0][0]),
            float(wicket_line[0][1]),
            src_w,
            src_h,
            scene_box_w,
            scene_box_h,
            pad=4,
        )
        p1 = _fit_point_to_box(
            float(wicket_line[1][0]),
            float(wicket_line[1][1]),
            src_w,
            src_h,
            scene_box_w,
            scene_box_h,
            pad=4,
        )
        p0 = (p0[0] + margin, p0[1] + scene_y0)
        p1 = (p1[0] + margin, p1[1] + scene_y0)
        cv2.line(canvas, p0, p1, (220, 220, 255), 4, cv2.LINE_AA)
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        cv2.putText(
            canvas,
            "Wicket line",
            (mid[0] + 8, mid[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        canvas,
        (margin, scene_y0),
        (margin + scene_box_w, scene_y0 + scene_box_h),
        (90, 90, 100),
        2,
    )

    # Right column: panels
    col_x0 = split_x + 8
    col_w = canvas_w - col_x0 - margin
    y = header_h + margin
    col_h_avail = canvas_h - header_h - footer_h - 2 * margin
    gap = 12
    n_panels = 4
    panel_h = (col_h_avail - gap * (n_panels - 1)) // n_panels

    pitch = str(api.get("pitch", "—"))
    impact = str(api.get("impact", "—"))
    wickets = str(api.get("wickets", "—"))
    decision = str(api.get("decision", "—"))
    reason = api.get("reason")

    rows = [
        ("PITCHING", pitch, _panel_color_for_value("pitch", pitch)),
        ("IMPACT", impact, _panel_color_for_value("impact", impact)),
        ("WICKETS", wickets, _panel_color_for_value("wickets", wickets)),
    ]

    for label, value, accent in rows:
        cv2.rectangle(
            canvas,
            (col_x0, y),
            (col_x0 + col_w, y + panel_h),
            (42, 40, 38),
            -1,
        )
        cv2.rectangle(
            canvas,
            (col_x0, y),
            (col_x0 + col_w, y + 36),
            accent,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (col_x0 + 12, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Value (wrapped roughly)
        val_y = y + 36 + 42
        cv2.putText(
            canvas,
            value[:32],
            (col_x0 + 14, val_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1.05,
            (250, 250, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (col_x0, y),
            (col_x0 + col_w, y + panel_h),
            accent,
            2,
        )
        y += panel_h + gap

    # Decision panel (taller last slot — use remaining space)
    rem_y = y
    rem_h = canvas_h - footer_h - margin - rem_y
    rem_h = max(rem_h, panel_h)
    bar_c, border_c = _decision_colors(decision)
    cv2.rectangle(
        canvas,
        (col_x0, rem_y),
        (col_x0 + col_w, rem_y + rem_h),
        (38, 36, 34),
        -1,
    )
    cv2.rectangle(
        canvas,
        (col_x0, rem_y),
        (col_x0 + col_w, rem_y + 40),
        bar_c,
        -1,
    )
    cv2.putText(
        canvas,
        "AI DECISION",
        (col_x0 + 12, rem_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        decision[:24],
        (col_x0 + 14, rem_y + 40 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.35,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if reason:
        rs = str(reason)[:70]
        cv2.putText(
            canvas,
            rs,
            (col_x0 + 14, rem_y + 40 + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        canvas,
        (col_x0, rem_y),
        (col_x0 + col_w, rem_y + rem_h),
        border_c,
        3,
    )

    # Footer
    cv2.rectangle(
        canvas,
        (0, canvas_h - footer_h),
        (canvas_w, canvas_h),
        (44, 40, 36),
        -1,
    )
    foot = "Trajectory & geometry in image — review is indicative (2D model)."
    cv2.putText(
        canvas,
        foot,
        (margin, canvas_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 170),
        1,
        cv2.LINE_AA,
    )

    return canvas

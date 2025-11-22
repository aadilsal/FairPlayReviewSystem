"""Compatibility wrapper: import the implementation from `detection`.

This file keeps the original module path. Implementation now lives in
`detection.ball_detector`.
"""

from detection.ball_detector import get_yolo_detector, detect_ball_on_frame, color_finder, hsv_vals

__all__ = ["get_yolo_detector", "detect_ball_on_frame", "color_finder", "hsv_vals"]
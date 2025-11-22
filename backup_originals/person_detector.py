"""Compatibility wrapper delegating to `detection.person_detector`.

Kept for backward compatibility with existing imports.
"""

from detection.person_detector import (
    detect_persons,
    iou,
    set_batsman_weights,
)

__all__ = ["detect_persons", "iou", "set_batsman_weights"]

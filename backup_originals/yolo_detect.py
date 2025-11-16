"""Compatibility wrapper delegating to `models.yolo_detect`.

Kept for backward compatibility; the implementation now lives in the
`models` package.
"""

from models.yolo_detect import YOLOBallDetector

__all__ = ["YOLOBallDetector"]

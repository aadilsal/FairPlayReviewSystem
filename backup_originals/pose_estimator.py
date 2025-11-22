"""Compatibility wrapper delegating to `detection.pose_detector`.

Kept for backward compatibility with existing imports.
"""

from detection.pose_detector import estimate_pose

__all__ = ["estimate_pose"]

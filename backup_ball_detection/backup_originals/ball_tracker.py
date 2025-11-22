"""Compatibility wrapper delegating to `detection.ball_tracker`.

Retained for backwards compatibility with scripts importing the top-level
module name.
"""

from detection.ball_tracker import ball_detect

__all__ = ["ball_detect"]

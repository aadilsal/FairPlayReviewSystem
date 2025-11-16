"""Compatibility wrapper delegating to `pipeline.postprocessing`.

Kept so existing imports of `frames_to_video_with_custom_path` continue to work.
"""

from pipeline.postprocessing import frames_to_video_with_custom_path

__all__ = ["frames_to_video_with_custom_path"]
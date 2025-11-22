"""Compatibility wrapper delegating to `pipeline.preprocessing.extract_video_frames`.

Kept so scripts that import `frame_extractor.extract_video_frames` continue to work.
"""

from pipeline.preprocessing import extract_video_frames

__all__ = ["extract_video_frames"]
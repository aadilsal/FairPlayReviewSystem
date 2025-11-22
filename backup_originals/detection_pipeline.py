"""Backward-compatible wrapper delegating to `pipeline.main_pipeline`.

This file preserves the original module path (`detection_pipeline`) so older
imports continue to work while the implementation lives in
`pipeline.main_pipeline`.
"""

from pipeline.main_pipeline import process_frames_pipeline

__all__ = ["process_frames_pipeline"]
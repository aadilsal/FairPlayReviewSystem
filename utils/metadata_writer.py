import json
from pathlib import Path

def save_metadata_for_frame(frame_path, metadata: dict):
    """
    Save per-frame metadata as JSON alongside the frame.
    Example metadata keys:
      - frame_index
      - persons: [[x,y,w,h], ...]
      - bats: [[x,y,w,h,conf], ...]
      - batsman_confirmed: bool
      - batsman_bbox: [x,y,w,h] (when confirmed)
      - tracked: True/False (when tracking)
    """
    try:
        meta_path = Path(frame_path).with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass
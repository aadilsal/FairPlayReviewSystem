# Weights folder (logical)

This folder is the logical destination for model weight files (checkpoints).

Notes:
- Large checkpoint files (e.g., `yolov8s.pt`) remain at the repository root by
  default to avoid accidental duplication during automated moves. You can
  manually move desired files into this folder and then update `config/paths.py`
  to point to them (or set the environment variables `YOLO_BALL_WEIGHTS`,
  `BATSMAN_WEIGHTS`, `POSE_WEIGHTS`).

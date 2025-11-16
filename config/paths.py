"""Centralized path and weights configuration.

This file maps logical names used across the codebase to actual file paths.
It intentionally points to the existing locations in the repository so large
weight files are not moved automatically.
"""
import os

# Helper directories
# `REPO_ROOT` is the repository root (one level up from this file)
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(REPO_ROOT, 'weights')

# We keep the large weight files in the `weights/` folder by default. If you
# prefer a different layout, set the environment variables listed below or
# update these paths.
#
# NOTE: The default ball detector weights were updated to the custom
# Roboflow RF-DETR export. If you have a different runtime path, set
# the `YOLO_BALL_WEIGHTS` env var or call `weights_config.set_yolo_ball_weights`.
YOLO_BALL_WEIGHTS = os.environ.get('YOLO_BALL_WEIGHTS', os.path.join(WEIGHTS_DIR, 'weights.pt'))
BATSMAN_WEIGHTS = os.environ.get('BATSMAN_WEIGHTS', os.path.join(WEIGHTS_DIR, 'yolov8s.pt'))
POSE_WEIGHTS = os.environ.get('POSE_WEIGHTS', os.path.join(WEIGHTS_DIR, 'yolov8s-pose.pt'))

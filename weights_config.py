"""Simple runtime weights configuration.

This module is a tiny mutable holder used by the pipeline to know which
weights to use for detectors. The training API will update this file's
content (in-memory) so services can switch to newly-trained weights.
"""
# Default paths (these can be updated at runtime)
YOLO_BALL_WEIGHTS = 'ball-yolov8s.pt'
BATSMAN_WEIGHTS = 'yolov8s.pt'


def set_yolo_ball_weights(path: str):
    global YOLO_BALL_WEIGHTS
    YOLO_BALL_WEIGHTS = path


def set_batsman_weights(path: str):
    global BATSMAN_WEIGHTS
    BATSMAN_WEIGHTS = path

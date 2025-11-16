"""Runtime weights configuration shim.

This module preserves the original `weights_config` API but delegates the
actual default paths to `config.paths`. Setters update the values in
`config.paths` so other modules that read from `config.paths` see the change.
"""
from config import paths as _paths


# Expose current defaults for backward compatibility
YOLO_BALL_WEIGHTS = _paths.YOLO_BALL_WEIGHTS
BATSMAN_WEIGHTS = _paths.BATSMAN_WEIGHTS


def set_yolo_ball_weights(path: str):
    """Set the YOLO ball weights path at runtime.

    Updates the centralized `config.paths` so all code reading that module
    sees the new value.
    """
    _paths.YOLO_BALL_WEIGHTS = path
    global YOLO_BALL_WEIGHTS
    YOLO_BALL_WEIGHTS = path


def set_batsman_weights(path: str):
    _paths.BATSMAN_WEIGHTS = path
    global BATSMAN_WEIGHTS
    BATSMAN_WEIGHTS = path

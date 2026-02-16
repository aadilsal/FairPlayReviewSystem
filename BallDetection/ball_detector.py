import cv2
import logging
import os
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import scipy.interpolate
from BallDetection.yolo_detect import yolo_detect_ball, get_global_yolo_detector
from BallDetection.filters import filter_and_select_ball_detection
from BallDetection.interpolation import interpolate_trajectory
from BallDetection.config import DETECTION_CONFIG

logger = logging.getLogger(__name__)

def detect_ball_on_frame(
    frame,
    yolo_weights=None,
    debug=False,
    ball_color=None,
    conf_threshold=None,
    iou_threshold=None,
    imgsz=None,
    frame_idx=0
):
    ball_color = DETECTION_CONFIG['ball_color']
    detector = get_global_yolo_detector(yolo_weights)
    yolo_detections = yolo_detect_ball(detector, frame)

    ball_info = filter_and_select_ball_detection(frame, yolo_detections, ball_color)
    if ball_info:
        logger.info(f"Frame {frame_idx}: Ball detected at ({ball_info['box'][0]}, {ball_info['box'][1]}) with conf {ball_info['conf']:.2f}")
    interpolated_positions = interpolate_trajectory([ball_info] if ball_info else [None])
    if ball_info:
        ball_info['interpolated_position'] = interpolated_positions[0]
    return frame, ball_info


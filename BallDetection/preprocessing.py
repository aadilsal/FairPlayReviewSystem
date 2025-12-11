"""Preprocessing utilities for cricket ball detection.
"""
import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def estimate_blur(frame: np.ndarray) -> float:
    """
    Estimate blur level using Laplacian variance (lower = blurrier)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_frame(
    frame: np.ndarray,
    ball_color: str = 'white',
    enable_deblur: bool = True,
    enable_sharpen: bool = True,
    enable_clahe: bool = True,
    blur_threshold: float = 100.0
) -> Tuple[np.ndarray, dict]:
    """
    Enhanced preprocessing for blurry/small ball detection.

    Returns (processed_frame, debug_info)
    """
    debug_info = {}
    processed = frame.copy()

    blur_score = estimate_blur(frame)
    debug_info['blur_score'] = float(blur_score)
    is_blurry = blur_score < blur_threshold
    debug_info['is_blurry'] = bool(is_blurry)

    # Deblur: bilateral filter (fast, edge-preserving)
    if enable_deblur and is_blurry:
        logger.debug(f"Frame blurry (score={blur_score:.1f}), applying bilateralFilter")
        processed = cv2.bilateralFilter(processed, d=5, sigmaColor=75, sigmaSpace=75)
        debug_info['deblur_applied'] = True

    # CLAHE on L channel
    if enable_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        debug_info['clahe_applied'] = True

    # Color-specific enhancements
    if ball_color == 'white':
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Boost brightness slightly
        v = cv2.add(v, 20)
        processed = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        debug_info['white_boost'] = True
    elif ball_color == 'red':
        # Slightly boost red channel
        processed[:, :, 2] = cv2.add(processed[:, :, 2], 15)
        debug_info['red_boost'] = True

    # Sharpen
    if enable_sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)
        debug_info['sharpen_applied'] = True

    return processed, debug_info


def preprocess_for_color_fallback(frame: np.ndarray, ball_color: str) -> np.ndarray:
    """Aggressive preprocessing specifically for color-based detection fallback."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    processed = cv2.merge([l, a, b])
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    return processed

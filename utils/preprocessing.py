import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def preprocess_frame(
    frame,
    color_mode=None,
    enable_deblur=True,
    enable_sharpen=True,
    enable_clahe=True,
    blur_threshold=100.0
):
    processed = frame.copy()
    blur_score = estimate_blur(frame)
    is_blurry = blur_score < blur_threshold
    if enable_deblur and is_blurry:
        processed = cv2.bilateralFilter(processed, d=5, sigmaColor=75, sigmaSpace=75)
    if enable_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if color_mode == 'white':
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 20)
        processed = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    elif color_mode == 'red':
        processed[:, :, 2] = cv2.add(processed[:, :, 2], 15)
    if enable_sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    return processed

"""
Frame Preprocessing Module for Robust Ball Detection

This module provides quality assessment and image enhancement functions to improve
ball detection under poor image quality conditions (brightness, blur, contrast issues).

Features:
- Quality assessment (brightness, blur, overall quality)
- Adaptive image enhancement (brightness normalization, CLAHE, sharpening)
- Performance monitoring and logging
- Optional preprocessing toggle

Author: FairPlayReviewSystem
Date: November 22, 2025
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrightnessLevel(Enum):
    """Brightness classification."""
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    NORMAL = "normal"


@dataclass
class QualityMetrics:
    """Container for frame quality metrics."""
    brightness_level: str
    brightness_value: float  # 0-1
    blur_score: float  # 0-1, higher = blurrier
    contrast_score: float  # 0-1
    overall_quality: float  # 0-1, higher = better
    needs_enhancement: bool


@dataclass
class PreprocessingResult:
    """Container for preprocessing results."""
    frame: np.ndarray
    was_preprocessed: bool
    quality_metrics: QualityMetrics
    processing_time_ms: float
    enhancements_applied: list


# ==================== QUALITY ASSESSMENT ====================

def detect_brightness_level(
    frame: np.ndarray,
    dark_threshold: float = 0.3,
    bright_threshold: float = 0.7
) -> str:
    """
    Detect brightness level of a frame.
    
    Args:
        frame: Input frame (BGR or grayscale)
        dark_threshold: Below this value (0-1) is considered too dark
        bright_threshold: Above this value (0-1) is considered too bright
    
    Returns:
        "too_dark", "too_bright", or "normal"
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate average brightness (normalized to 0-1)
    avg_brightness = np.mean(gray) / 255.0
    
    if avg_brightness < dark_threshold:
        return BrightnessLevel.TOO_DARK.value
    elif avg_brightness > bright_threshold:
        return BrightnessLevel.TOO_BRIGHT.value
    else:
        return BrightnessLevel.NORMAL.value


def detect_blur_level(frame: np.ndarray) -> float:
    """
    Detect blur level using Laplacian variance method.
    
    Args:
        frame: Input frame (BGR or grayscale)
    
    Returns:
        blur_score: 0-1, where 0 is sharp and 1 is very blurry
        
    Note:
        - Uses Laplacian variance to detect edges
        - Lower variance = more blur
        - Normalized to 0-1 range for convenience
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize to 0-1 scale (inverse, so 1 = blurry)
    # Typical variance range: 0-1000+ for sharp images
    # Using sigmoid-like normalization
    max_variance = 500.0  # Tunable threshold
    blur_score = 1.0 - min(variance / max_variance, 1.0)
    
    return blur_score


def calculate_contrast_score(frame: np.ndarray) -> float:
    """
    Calculate contrast score using standard deviation.
    
    Args:
        frame: Input frame (BGR or grayscale)
    
    Returns:
        contrast_score: 0-1, where higher = better contrast
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Standard deviation as contrast measure
    std_dev = np.std(gray)
    
    # Normalize to 0-1 (typical std_dev range: 0-70)
    contrast_score = min(std_dev / 70.0, 1.0)
    
    return contrast_score


def get_frame_quality_score(
    frame: np.ndarray,
    brightness_weight: float = 0.3,
    blur_weight: float = 0.4,
    contrast_weight: float = 0.3
) -> Tuple[float, QualityMetrics]:
    """
    Get overall frame quality score.
    
    Args:
        frame: Input frame (BGR or grayscale)
        brightness_weight: Weight for brightness component
        blur_weight: Weight for blur component
        contrast_weight: Weight for contrast component
    
    Returns:
        Tuple of (overall_quality_score, QualityMetrics object)
        quality_score: 0-1, where higher = better quality
    """
    # Get individual metrics
    brightness_level = detect_brightness_level(frame)
    blur_score = detect_blur_level(frame)
    contrast_score = calculate_contrast_score(frame)
    
    # Calculate brightness value for scoring
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    brightness_value = np.mean(gray) / 255.0
    
    # Score brightness (optimal around 0.5)
    brightness_score = 1.0 - abs(brightness_value - 0.5) * 2.0
    brightness_score = max(0.0, brightness_score)
    
    # Score blur (lower blur_score = better)
    sharpness_score = 1.0 - blur_score
    
    # Calculate weighted overall quality
    overall_quality = (
        brightness_weight * brightness_score +
        blur_weight * sharpness_score +
        contrast_weight * contrast_score
    )
    
    # Determine if enhancement is needed
    needs_enhancement = (
        overall_quality < 0.6 or
        brightness_level != BrightnessLevel.NORMAL.value or
        blur_score > 0.5 or
        contrast_score < 0.4
    )
    
    metrics = QualityMetrics(
        brightness_level=brightness_level,
        brightness_value=brightness_value,
        blur_score=blur_score,
        contrast_score=contrast_score,
        overall_quality=overall_quality,
        needs_enhancement=needs_enhancement
    )
    
    return overall_quality, metrics


# ==================== IMAGE ENHANCEMENT ====================

def normalize_brightness(
    frame: np.ndarray,
    target_brightness: float = 0.5,
    method: str = "gamma"
) -> np.ndarray:
    """
    Normalize brightness to target level.
    
    Args:
        frame: Input frame (BGR)
        target_brightness: Desired brightness (0-1)
        method: "gamma" or "linear"
    
    Returns:
        Enhanced frame with normalized brightness
    """
    # Convert to grayscale to measure brightness
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    current_brightness = np.mean(gray) / 255.0
    
    if abs(current_brightness - target_brightness) < 0.05:
        return frame.copy()  # Already close to target
    
    if method == "gamma":
        # Gamma correction
        gamma = np.log(target_brightness) / np.log(current_brightness + 1e-6)
        gamma = np.clip(gamma, 0.3, 3.0)  # Limit extreme corrections
        
        # Apply gamma correction
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 
                                for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(frame, lookup_table)
        
    else:  # linear
        # Linear scaling
        alpha = target_brightness / (current_brightness + 1e-6)
        alpha = np.clip(alpha, 0.5, 2.0)  # Limit extreme corrections
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    return enhanced


def enhance_contrast(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        frame: Input frame (BGR)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast-enhanced frame
    """
    # Convert to LAB color space for better results
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_enhanced = clahe.apply(l_channel)
    
    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_channel_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def reduce_blur(
    frame: np.ndarray,
    strength: float = 1.0,
    method: str = "unsharp"
) -> np.ndarray:
    """
    Reduce blur using sharpening techniques.
    
    Args:
        frame: Input frame (BGR)
        strength: Sharpening strength (0.5-2.0, where 1.0 is normal)
        method: "unsharp" for unsharp masking or "kernel" for kernel-based
    
    Returns:
        Sharpened frame
    """
    strength = np.clip(strength, 0.5, 2.0)
    
    if method == "unsharp":
        # Unsharp masking: original + (original - blurred) * strength
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + strength, gaussian, -strength, 0)
        
    else:  # kernel
        # Sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32) * (strength / 1.0)
        kernel[1, 1] = 5  # Center value remains constant
        
        sharpened = cv2.filter2D(frame, -1, kernel)
    
    return sharpened


def preprocess_frame(
    frame: np.ndarray,
    target_brightness: float = 0.5,
    apply_clahe: bool = True,
    apply_sharpening: bool = True,
    sharpening_strength: float = 1.0,
    adaptive: bool = True
) -> Tuple[np.ndarray, list]:
    """
    Apply all enhancements intelligently based on frame quality.
    
    Args:
        frame: Input frame (BGR)
        target_brightness: Target brightness for normalization
        apply_clahe: Whether to apply CLAHE contrast enhancement
        apply_sharpening: Whether to apply sharpening
        sharpening_strength: Strength of sharpening (0.5-2.0)
        adaptive: If True, only apply enhancements where needed
    
    Returns:
        Tuple of (enhanced_frame, list_of_applied_enhancements)
    """
    enhanced = frame.copy()
    applied_enhancements = []
    
    if adaptive:
        # Assess quality to decide which enhancements to apply
        _, metrics = get_frame_quality_score(frame)
        
        # Brightness normalization
        if metrics.brightness_level != BrightnessLevel.NORMAL.value:
            enhanced = normalize_brightness(enhanced, target_brightness)
            applied_enhancements.append("brightness_normalization")
        
        # Contrast enhancement
        if apply_clahe and metrics.contrast_score < 0.5:
            enhanced = enhance_contrast(enhanced)
            applied_enhancements.append("clahe_contrast")
        
        # Sharpening
        if apply_sharpening and metrics.blur_score > 0.4:
            enhanced = reduce_blur(enhanced, strength=sharpening_strength)
            applied_enhancements.append("sharpening")
    
    else:
        # Apply all enhancements regardless
        enhanced = normalize_brightness(enhanced, target_brightness)
        applied_enhancements.append("brightness_normalization")
        
        if apply_clahe:
            enhanced = enhance_contrast(enhanced)
            applied_enhancements.append("clahe_contrast")
        
        if apply_sharpening:
            enhanced = reduce_blur(enhanced, strength=sharpening_strength)
            applied_enhancements.append("sharpening")
    
    return enhanced, applied_enhancements


# ==================== ADAPTIVE DETECTION WRAPPER ====================

class AdaptiveBallDetector:
    """
    Wrapper for ball detection with adaptive preprocessing.
    
    This class integrates quality assessment and preprocessing with YOLO detection,
    adjusting confidence thresholds based on frame quality.
    """
    
    def __init__(
        self,
        model,
        enable_preprocessing: bool = True,
        quality_threshold: float = 0.6,
        base_confidence: float = 0.25,
        min_confidence: float = 0.15,
        target_brightness: float = 0.5,
        log_enhancements: bool = True
    ):
        """
        Initialize adaptive ball detector.
        
        Args:
            model: YOLO model instance
            enable_preprocessing: Enable/disable preprocessing
            quality_threshold: Quality below this triggers preprocessing
            base_confidence: Base YOLO confidence threshold
            min_confidence: Minimum confidence for low-quality frames
            target_brightness: Target brightness for normalization
            log_enhancements: Whether to log enhancement operations
        """
        self.model = model
        self.enable_preprocessing = enable_preprocessing
        self.quality_threshold = quality_threshold
        self.base_confidence = base_confidence
        self.min_confidence = min_confidence
        self.target_brightness = target_brightness
        self.log_enhancements = log_enhancements
        
        # Statistics
        self.stats = {
            "total_frames": 0,
            "preprocessed_frames": 0,
            "total_preprocessing_time_ms": 0.0,
            "quality_scores": []
        }
    
    def detect(
        self,
        frame: np.ndarray,
        frame_id: Optional[str] = None,
        imgsz: int = 640,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Run adaptive ball detection on a frame.
        
        Args:
            frame: Input frame (BGR)
            frame_id: Optional frame identifier for logging
            imgsz: YOLO inference size
            device: 'cpu' or 'cuda'
        
        Returns:
            Dictionary containing:
                - boxes: List of [x1, y1, x2, y2] bounding boxes
                - confidences: List of confidence scores
                - class_ids: List of class IDs
                - class_names: List of class names
                - quality_score: Overall quality score (0-1)
                - was_preprocessed: Boolean indicating if preprocessing was applied
                - preprocessing_time_ms: Time spent on preprocessing
                - enhancements_applied: List of enhancement operations
                - confidence_threshold: Adjusted confidence threshold used
        """
        start_time = time.time()
        
        # Step 1: Assess frame quality
        quality_score, quality_metrics = get_frame_quality_score(frame)
        
        # Step 2: Decide if preprocessing is needed
        needs_preprocessing = (
            self.enable_preprocessing and 
            quality_metrics.needs_enhancement
        )
        
        preprocessed_frame = frame
        enhancements_applied = []
        preprocessing_time_ms = 0.0
        
        if needs_preprocessing:
            preprocess_start = time.time()
            
            # Apply preprocessing
            preprocessed_frame, enhancements_applied = preprocess_frame(
                frame,
                target_brightness=self.target_brightness,
                apply_clahe=True,
                apply_sharpening=True,
                adaptive=True
            )
            
            preprocessing_time_ms = (time.time() - preprocess_start) * 1000
            
            # Update statistics
            self.stats["preprocessed_frames"] += 1
            self.stats["total_preprocessing_time_ms"] += preprocessing_time_ms
            
            if self.log_enhancements and enhancements_applied:
                log_msg = f"Frame {frame_id or 'unknown'}: Applied {', '.join(enhancements_applied)}"
                log_msg += f" (quality: {quality_score:.2f}, time: {preprocessing_time_ms:.1f}ms)"
                logger.info(log_msg)
        
        # Step 3: Adjust confidence threshold based on quality
        adjusted_confidence = self._adjust_confidence_threshold(quality_score)
        
        # Step 4: Run YOLO detection
        results = self.model.predict(
            source=preprocessed_frame,
            imgsz=imgsz,
            conf=adjusted_confidence,
            device=device,
            verbose=False
        )
        
        # Step 5: Parse results
        boxes = []
        confidences = []
        class_ids = []
        class_names = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append(xyxy.tolist())
                    
                    # Extract confidence
                    conf = float(box.conf[0].cpu().numpy())
                    confidences.append(conf)
                    
                    # Extract class
                    cls_id = int(box.cls[0].cpu().numpy())
                    class_ids.append(cls_id)
                    
                    # Get class name
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    class_names.append(cls_name)
        
        # Update statistics
        self.stats["total_frames"] += 1
        self.stats["quality_scores"].append(quality_score)
        
        # Return comprehensive results
        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "class_names": class_names,
            "quality_score": quality_score,
            "quality_metrics": quality_metrics,
            "was_preprocessed": needs_preprocessing,
            "preprocessing_time_ms": preprocessing_time_ms,
            "enhancements_applied": enhancements_applied,
            "confidence_threshold": adjusted_confidence,
            "total_time_ms": (time.time() - start_time) * 1000
        }
    
    def _adjust_confidence_threshold(self, quality_score: float) -> float:
        """
        Adjust confidence threshold based on frame quality.
        
        Lower quality frames use lower confidence thresholds to avoid
        missing detections in challenging conditions.
        
        Args:
            quality_score: Frame quality score (0-1)
        
        Returns:
            Adjusted confidence threshold
        """
        if quality_score >= 0.7:
            # High quality: use base confidence
            return self.base_confidence
        elif quality_score >= 0.5:
            # Medium quality: slightly lower confidence
            return self.base_confidence * 0.9
        else:
            # Low quality: use minimum confidence
            return max(self.min_confidence, self.base_confidence * 0.7)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        avg_quality = (
            sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
            if self.stats["quality_scores"] else 0.0
        )
        
        preprocessing_rate = (
            self.stats["preprocessed_frames"] / self.stats["total_frames"]
            if self.stats["total_frames"] > 0 else 0.0
        )
        
        avg_preprocessing_time = (
            self.stats["total_preprocessing_time_ms"] / self.stats["preprocessed_frames"]
            if self.stats["preprocessed_frames"] > 0 else 0.0
        )
        
        return {
            "total_frames_processed": self.stats["total_frames"],
            "frames_preprocessed": self.stats["preprocessed_frames"],
            "preprocessing_rate": preprocessing_rate,
            "average_quality_score": avg_quality,
            "average_preprocessing_time_ms": avg_preprocessing_time,
            "total_preprocessing_time_ms": self.stats["total_preprocessing_time_ms"]
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "total_frames": 0,
            "preprocessed_frames": 0,
            "total_preprocessing_time_ms": 0.0,
            "quality_scores": []
        }


# ==================== UTILITY FUNCTIONS ====================

def visualize_quality_assessment(
    frame: np.ndarray,
    quality_metrics: QualityMetrics,
    show_metrics: bool = True
) -> np.ndarray:
    """
    Visualize quality metrics on frame.
    
    Args:
        frame: Input frame (BGR)
        quality_metrics: QualityMetrics object
        show_metrics: Whether to overlay metrics text
    
    Returns:
        Frame with quality visualization
    """
    vis_frame = frame.copy()
    
    if show_metrics:
        # Prepare text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        texts = [
            f"Quality: {quality_metrics.overall_quality:.2f}",
            f"Brightness: {quality_metrics.brightness_level} ({quality_metrics.brightness_value:.2f})",
            f"Blur: {quality_metrics.blur_score:.2f}",
            f"Contrast: {quality_metrics.contrast_score:.2f}",
            f"Needs Enhancement: {quality_metrics.needs_enhancement}"
        ]
        
        # Draw background rectangle
        cv2.rectangle(vis_frame, (10, 10), (400, y_offset * len(texts) + 10), 
                     (0, 0, 0), -1)
        
        # Draw text
        for i, text in enumerate(texts):
            y = y_offset * (i + 1)
            cv2.putText(vis_frame, text, (15, y), font, font_scale, 
                       (0, 255, 0), thickness, cv2.LINE_AA)
    
    return vis_frame


def compare_preprocessing(
    frame: np.ndarray,
    show_original: bool = True,
    show_enhanced: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compare original and preprocessed frames side by side.
    
    Args:
        frame: Input frame (BGR)
        show_original: Include original frame
        show_enhanced: Include enhanced frame
    
    Returns:
        Dictionary with 'original', 'enhanced', and 'comparison' frames
    """
    results = {}
    
    if show_original:
        results["original"] = frame.copy()
    
    if show_enhanced:
        enhanced, enhancements = preprocess_frame(frame, adaptive=True)
        results["enhanced"] = enhanced
        results["enhancements_applied"] = enhancements
    
    if show_original and show_enhanced:
        # Create side-by-side comparison
        h, w = frame.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = frame
        comparison[:, w:] = results["enhanced"]
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Enhanced", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        results["comparison"] = comparison
    
    return results

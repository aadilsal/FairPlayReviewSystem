"""
Wicket Geometry Module

Resolves wicket positions from detections with fallback to configuration.
Priority: Manual config > Auto-detection > Defaults
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("fairplay.wicket_geometry")


def resolve_wicket_positions(
    det_wickets: List[List[Dict[str, Any]]],
    wicket_config: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[float, float, float, float]]:
    """
    Resolve wicket positions from detections, with config fallback.
    
    Args:
        det_wickets: Per-frame wicket detections
        wicket_config: Optional manual config with keys like 'wicket_near_x', 'wicket_far_x', etc.
    
    Returns:
        (x_near, y_base, x_far, y_base) or None
    """
    
    # Priority 1: Manual configuration
    if wicket_config:
        if all(k in wicket_config for k in ['wicket_near_x', 'wicket_far_x', 'wicket_y_base']):
            x_near = float(wicket_config['wicket_near_x'])
            x_far = float(wicket_config['wicket_far_x'])
            y_base = float(wicket_config['wicket_y_base'])
            logger.info(f"Using manual wicket config: near={x_near}, far={x_far}, y={y_base}")
            return (x_near, y_base, x_far, y_base)
    
    # Priority 2: Auto-detection from frame detections
    wicket_positions = _extract_wickets_from_detections(det_wickets)
    if wicket_positions:
        logger.info(f"Using auto-detected wickets: {wicket_positions}")
        return wicket_positions
    
    # Priority 3: Defaults (frame dimensions assumed)
    logger.warning("Using default wicket positions (no detection or config)")
    return None


def _extract_wickets_from_detections(
    det_wickets: List[List[Dict[str, Any]]]
) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract wicket positions from per-frame detections.
    Looks for labels like "Wicket_Near" and "Wicket_Far".
    """
    wicket_near_xs = []
    wicket_far_xs = []
    wicket_ys = []
    
    for frame_wickets in det_wickets:
        for det in frame_wickets:
            label = det.get("label", "")
            box = det.get("box", [])
            
            if len(box) < 4:
                continue
            
            x, y, w, h = box
            wicket_x = x + w / 2.0  # Center x
            wicket_y = y + h  # Bottom y
            
            if "Near" in label or "near" in label:
                wicket_near_xs.append(wicket_x)
            elif "Far" in label or "far" in label:
                wicket_far_xs.append(wicket_x)
            
            wicket_ys.append(wicket_y)
    
    if not wicket_near_xs or not wicket_far_xs or not wicket_ys:
        return None
    
    x_near = float(np.median(wicket_near_xs))
    x_far = float(np.median(wicket_far_xs))
    y_base = float(np.median(wicket_ys))
    
    return (x_near, y_base, x_far, y_base)


def get_default_wicket_positions(frame_width: int, frame_height: int) -> Tuple[float, float, float, float]:
    """
    Return default wicket positions based on frame dimensions.
    Assumes wickets are centered horizontally, at frame bottom.
    """
    wicket_spacing = frame_width / 6.0  # 1/6 of width apart
    center_x = frame_width / 2.0
    
    x_near = center_x - wicket_spacing / 2.0
    x_far = center_x + wicket_spacing / 2.0
    y_base = float(frame_height - 50)  # 50 pixels from bottom
    
    return (x_near, y_base, x_far, y_base)


def is_vertical_alignment(x_near: float, x_far: float, tolerance: float = 10.0) -> bool:
    """Check if wickets are approximately vertically aligned."""
    return abs(x_near - x_far) <= tolerance

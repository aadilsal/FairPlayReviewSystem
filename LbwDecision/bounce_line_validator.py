"""
Bounce Line Validator Module

Validates whether ball pitched/impacted in-line with wicket line.
"""

import logging
from typing import Tuple

logger = logging.getLogger("fairplay.bounce_line_validator")


def is_pitch_in_line(
    pitch_x: float,
    wicket_x_near: float,
    wicket_x_far: float,
    tolerance_px: float = 30.0
) -> bool:
    """
    Check if pitch point X-coordinate is between wickets.
    
    Args:
        pitch_x: X-coordinate of pitch point (bounce)
        wicket_x_near, wicket_x_far: X-coordinates of two stumps
        tolerance_px: Tolerance in pixels (±)
    
    Returns:
        True if pitch is in-line with wickets
    """
    x_min = min(wicket_x_near, wicket_x_far) - tolerance_px
    x_max = max(wicket_x_near, wicket_x_far) + tolerance_px
    
    in_line = x_min <= pitch_x <= x_max
    
    logger.debug(f"Pitch in-line check: pitch_x={pitch_x:.1f}, range=[{x_min:.1f}, {x_max:.1f}], result={in_line}")
    
    return in_line


def is_impact_in_line(
    impact_x: float,
    wicket_x_near: float,
    wicket_x_far: float,
    tolerance_px: float = 30.0
) -> bool:
    """
    Check if impact point X-coordinate is between wickets.
    
    Args:
        impact_x: X-coordinate of impact point
        wicket_x_near, wicket_x_far: X-coordinates of two stumps
        tolerance_px: Tolerance in pixels (±)
    
    Returns:
        True if impact is in-line with wickets
    """
    x_min = min(wicket_x_near, wicket_x_far) - tolerance_px
    x_max = max(wicket_x_near, wicket_x_far) + tolerance_px
    
    in_line = x_min <= impact_x <= x_max
    
    logger.debug(f"Impact in-line check: impact_x={impact_x:.1f}, range=[{x_min:.1f}, {x_max:.1f}], result={in_line}")
    
    return in_line


def check_straight_line(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    tolerance_y: float = 5.0
) -> bool:
    """
    Check if two points form an approximately horizontal line (Y-values close).
    Sanity check for wicket lines.
    
    Args:
        p1, p2: (x, y) coordinates
        tolerance_y: Max Y-distance for "horizontal"
    
    Returns:
        True if points are approximately horizontally aligned
    """
    _, y1 = p1
    _, y2 = p2
    
    is_horizontal = abs(y1 - y2) <= tolerance_y
    
    logger.debug(f"Straight line check: y1={y1:.1f}, y2={y2:.1f}, diff={abs(y1-y2):.1f}, is_horizontal={is_horizontal}")
    
    return is_horizontal

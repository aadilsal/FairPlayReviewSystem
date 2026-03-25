import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from BallDetection.utils.config import POST_PROCESSOR_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class GapInfo:
    start_frame: int
    end_frame: int
    gap_type: str  # 'mid_flight', 'bounce_adjacent', 'occlusion'
    pre_anchor: Optional[Dict[str, Any]]
    post_anchor: Optional[Dict[str, Any]]

def classify_gaps(ball_infos: List[Optional[Dict[str, Any]]]) -> List[GapInfo]:
    """
    Iterates over collected ball_info dicts and identifies contiguous runs of None or ghost frames.
    For each gap, classifies it as mid_flight, bounce_adjacent, or occlusion.
    """
    gaps = []
    n = len(ball_infos)
    i = 0
    
    while i < n:
        # A gap is defined as a None entry or an entry marked as 'ghost'
        is_current_gap = ball_infos[i] is None or ball_infos[i].get('ghost', False)
        
        if is_current_gap:
            start_frame = i
            while i < n and (ball_infos[i] is None or ball_infos[i].get('ghost', False)):
                i += 1
            end_frame = i - 1
            
            # Locate the nearest non-ghost anchors surrounding the gap
            pre_anchor = None
            for j in range(start_frame - 1, -1, -1):
                if ball_infos[j] and not ball_infos[j].get('ghost', False):
                    pre_anchor = ball_infos[j]
                    break
            
            post_anchor = None
            for j in range(end_frame + 1, n):
                if ball_infos[j] and not ball_infos[j].get('ghost', False):
                    post_anchor = ball_infos[j]
                    break
            
            # Determine gap type using heuristics
            gap_type = _classify_gap_type(ball_infos, start_frame, end_frame, pre_anchor, post_anchor)
            
            gaps.append(GapInfo(
                start_frame=start_frame,
                end_frame=end_frame,
                gap_type=gap_type,
                pre_anchor=pre_anchor,
                post_anchor=post_anchor
            ))
        else:
            i += 1
            
    return gaps

def _classify_gap_type(ball_infos: List[Any], start: int, end: int, 
                       pre: Optional[Dict], post: Optional[Dict]) -> str:
    """
    Heuristic-based classification of ball detection gaps.
    
    1. occlusion: If anchors exist but ball was near typical bat/pad region 
       or confidence was dropping before the gap.
    2. bounce_adjacent: If vy flips sign across the gap.
    3. mid_flight: Baseline state (default).
    """
    if not pre or not post:
        return 'mid_flight'
    
    # 1. Occlusion Heuristics
    # Check if confidence was dropping significantly just before the gap
    conf_trend = []
    for j in range(start - 1, -1, -1):
        if ball_infos[j] and not ball_infos[j].get('ghost', False):
            conf_trend.append(ball_infos[j].get('conf', 0))
            if len(conf_trend) >= 3:
                break
    
    if len(conf_trend) >= 2:
        # If current confidence (conf_trend[0]) is lower than previous (signifying ball becoming blurry or obscured)
        if conf_trend[0] < (conf_trend[1] - 0.1):
            return 'occlusion'
            
    # Proximity heuristic: If x-pos is in a central band where batsmen usually are
    # Note: 'known bat zone' would ideally come from external detection, 
    # but we can use a basic center-frame heuristic if needed. 

    # 2. Bounce Heuristic: vy flips sign across the gap
    vy_in = _get_velocity_at(ball_infos, start - 1, direction=-1)
    vy_out = _get_velocity_at(ball_infos, end + 1, direction=1)
    
    if vy_in is not None and vy_out is not None:
        # Check if v_y sign flipped (e.g., falling vs rising)
        if (vy_in > 0 and vy_out < 0) or (vy_in < 0 and vy_out > 0):
            # Verify magnitude of change exceeds threshold (prevents noise triggered flips)
            if abs(vy_in - vy_out) > POST_PROCESSOR_CONFIG.get('BOUNCE_VY_FLIP_THRESHOLD', 5.0):
                return 'bounce_adjacent'

    return 'mid_flight'

def _get_velocity_at(ball_infos: List[Any], idx: int, direction: int) -> Optional[float]:
    """Estimates vertical velocity (v_y) at a given index by looking for the nearest valid anchor."""
    curr = ball_infos[idx]
    if not curr or curr.get('ghost'):
        return None
    
    # Find next/prev valid anchor to compute displacement
    other = None
    step = 1 if direction > 0 else -1
    for j in range(idx + step, len(ball_infos) if direction > 0 else -1, step):
        if ball_infos[j] and not ball_infos[j].get('ghost', False):
            other = ball_infos[j]
            break
            
    if other:
        # Calculated as (delta_y / delta_frame)
        # Using [1] for y-coordinate of 'interpolated_position' tuple
        dy = other['interpolated_position'][1] - curr['interpolated_position'][1]
        dt = other['frame_idx'] - curr['frame_idx']
        return dy / dt if dt != 0 else 0
    
    return None

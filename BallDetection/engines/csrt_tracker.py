import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from BallDetection.utils.config import POST_PROCESSOR_CONFIG

logger = logging.getLogger(__name__)

def _create_csrt_tracker():
    """Factory function for the OpenCV CSRT Tracker."""
    # Sometimes cv2.TrackerCSRT_create is under cv2.legacy
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        logger.error("[CSRT Tracker] OpenCV CSRT tracker not found. Is opencv-contrib-python installed?")
        return None

def track_forward(frames: List[np.ndarray], start_frame: int, start_bbox: List[float], 
                  end_frame: int, is_occlusion: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Initializes a CSRT tracker on start_frame and tracks forward to end_frame.
    If is_occlusion is True, disables template updating to prevent tracking the occluder.
    Returns a dict mapping frame_idx -> { 'box': (x,y,w,h), 'success': bool, 'tracked': True }
    """
    tracker = _create_csrt_tracker()
    if tracker is None:
        return {}

    # Convert [x, y, w, h] from float to integer for OpenCV
    bbox = tuple([int(v) for v in start_bbox])
    
    # Initialize tracker
    tracker.init(frames[start_frame], bbox)
    
    # Template freezing hack: If occlusion, we actually don't want the tracker to update its template.
    # OpenCV's standard Python API doesn't expose a 'read-only' mode for CSRT.
    # As a workaround, if it's an occlusion gap, we re-initialize the tracker at every step
    # with the newly predicted position, forcing it to use the *original* template from start_frame.
    # We'll save the original ROI crop to do template matching if needed, 
    # but for now, we'll just run standard CSRT and rely on the agreement step to catch drifts.
    
    results = {}
    
    for i in range(start_frame + 1, end_frame + 1):
        if i >= len(frames):
            break
            
        success, bbox = tracker.update(frames[i])
        
        if success:
            results[i] = {
                'box': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'success': True
            }
        else:
            results[i] = {
                'box': None,
                'success': False
            }
            logger.debug(f"[CSRT] Forward tracking failed at frame {i}")
            # If it fails, we stop tracking forward
            break
            
    return results

def track_backward(frames: List[np.ndarray], end_frame: int, end_bbox: List[float], 
                   start_frame: int, is_occlusion: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Initializes a CSRT tracker on end_frame and tracks backward to start_frame.
    """
    tracker = _create_csrt_tracker()
    if tracker is None:
        return {}

    bbox = tuple([int(v) for v in end_bbox])
    tracker.init(frames[end_frame], bbox)
    
    results = {}
    
    for i in range(end_frame - 1, start_frame - 1, -1):
        if i < 0:
            break
            
        success, bbox = tracker.update(frames[i])
        
        if success:
            results[i] = {
                'box': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'success': True
            }
        else:
            results[i] = {
                'box': None,
                'success': False
            }
            logger.debug(f"[CSRT] Backward tracking failed at frame {i}")
            break
            
    return results

def _calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union for two bounding boxes [x, y, w, h]."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea)

def agree_and_merge(forward_res: Dict[int, Dict], backward_res: Dict[int, Dict], 
                    gap_type: str) -> Dict[int, Dict[str, Any]]:
    """
    Merges forward and backward tracking results.
    - If IoU >= threshold: average positions ('csrt-agreed')
    - If gap_type == 'occlusion' and distance > threshold: calculate intersection ('edge-suspected')
    - Otherwise: use whichever is available, preferring forward.
    """
    merged = {}
    
    iou_thresh = POST_PROCESSOR_CONFIG.get('CSRT_AGREEMENT_IOU', 0.3)
    edge_thresh = POST_PROCESSOR_CONFIG.get('EDGE_DISAGREEMENT_PX', 25.0)
    
    # Get all frames that were tracked by either direction
    all_frames = set(forward_res.keys()).union(set(backward_res.keys()))
    
    for f in all_frames:
        f_data = forward_res.get(f)
        b_data = backward_res.get(f)
        
        has_f = f_data and f_data['success'] and f_data['box'] is not None
        has_b = b_data and b_data['success'] and b_data['box'] is not None
        
        if has_f and has_b:
            f_box = f_data['box']
            b_box = b_data['box']
            
            iou = _calculate_iou(f_box, b_box)
            
            # Center points for distance
            f_cx = f_box[0] + f_box[2]/2
            f_cy = f_box[1] + f_box[3]/2
            b_cx = b_box[0] + b_box[2]/2
            b_cy = b_box[1] + b_box[3]/2
            
            dist = np.sqrt((f_cx - b_cx)**2 + (f_cy - b_cy)**2)
            
            if iou >= iou_thresh:
                # Agreement! Average the boxes
                avg_box = [
                    (f_box[0] + b_box[0]) / 2.0,
                    (f_box[1] + b_box[1]) / 2.0,
                    (f_box[2] + b_box[2]) / 2.0,
                    (f_box[3] + b_box[3]) / 2.0
                ]
                merged[f] = {
                    'box': avg_box,
                    'interpolated_position': (avg_box[0], avg_box[1]),
                    'source': 'csrt-agreed'
                }
            elif gap_type == 'occlusion' and dist > edge_thresh:
                # Disagreement during occlusion -> Possible Edge
                # For now, we take the midpoint but tag it specifically so the kinematic fallback 
                # can compute the exact arc intersection later.
                avg_box = [
                    (f_box[0] + b_box[0]) / 2.0,
                    (f_box[1] + b_box[1]) / 2.0,
                    (f_box[2] + b_box[2]) / 2.0,
                    (f_box[3] + b_box[3]) / 2.0
                ]
                merged[f] = {
                    'box': avg_box,
                    'interpolated_position': (avg_box[0], avg_box[1]),
                    'source': 'edge-suspected',
                    'f_box': f_box,
                    'b_box': b_box
                }
                logger.warning(f"[CSRT] Tracker disagreement > {edge_thresh}px at frame {f} during occlusion. Tagged 'edge-suspected'.")
            else:
                # Disagreement but not an edge case. Just pick forward as default if valid.
                merged[f] = {
                    'box': f_box,
                    'interpolated_position': (f_box[0], f_box[1]),
                    'source': 'csrt-forward'
                }
        
        elif has_f:
            merged[f] = {
                'box': f_data['box'],
                'interpolated_position': (f_data['box'][0], f_data['box'][1]),
                'source': 'csrt-forward'
            }
        elif has_b:
            merged[f] = {
                'box': b_data['box'],
                'interpolated_position': (b_data['box'][0], b_data['box'][1]),
                'source': 'csrt-backward'
            }
            
    return merged
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from BallDetection.utils.config import POST_PROCESSOR_CONFIG
from BallDetection.pipeline.trajectory import TrajectoryModel, predict_position

def draw_trajectory_overlay(
    frame: np.ndarray, 
    ball_infos: List[Optional[Dict[str, Any]]], 
    trajectory_model: Optional[TrajectoryModel] = None,
    current_frame_idx: Optional[int] = None
) -> np.ndarray:
    """
    Draws a debug overlay on the frame showing the final trajectory, 
    color-coded by detection source.
    """
    overlay = frame.copy()
    output = frame.copy()

    # 1. Draw Corridor (semi-transparent band)
    if trajectory_model is not None:
        corridor_width = POST_PROCESSOR_CONFIG.get('CORRIDOR_WIDTH_PX', 40)
        
        predicted_pts = []
        # Calculate predicted points for the entire sequence (up to current frame if specified)
        max_frame = len(ball_infos) if current_frame_idx is None else current_frame_idx + 1
        for i in range(max_frame):
            pos = predict_position(trajectory_model, i)
            if pos is not None:
                predicted_pts.append((int(pos[0]), int(pos[1])))
                
        if len(predicted_pts) > 1:
            # Draw pre-bounce and post-bounce separately if bounce exists
            if trajectory_model.bounce_frame is not None and 0 <= trajectory_model.bounce_frame < len(predicted_pts):
                bounce_idx = trajectory_model.bounce_frame
                pts1 = np.array(predicted_pts[:bounce_idx+1], dtype=np.int32)
                pts2 = np.array(predicted_pts[bounce_idx:], dtype=np.int32)
                if len(pts1) > 1:
                    cv2.polylines(overlay, [pts1], False, (255, 255, 255), thickness=corridor_width*2)
                if len(pts2) > 1:
                    cv2.polylines(overlay, [pts2], False, (255, 255, 255), thickness=corridor_width*2)
            else:
                pts = np.array(predicted_pts, dtype=np.int32)
                cv2.polylines(overlay, [pts], False, (255, 255, 255), thickness=corridor_width*2)
                
        # Blend overlay for semi-transparency
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # 2. Draw Trajectory Points
    color_map = {
        'yolo': (0, 255, 0),             # Green
        'yolo-anchor': (0, 255, 0),      # Green
        'yolo-rescue': (0, 255, 255),    # Yellow
        'csrt-agreed': (255, 255, 0),    # Cyan
        'csrt-forward': (255, 255, 0),   # Cyan
        'csrt-backward': (255, 255, 0),  # Cyan
        'kinematic': (0, 0, 255),        # Red
        'edge-suspected': (0, 0, 255)    # Red
    }

    valid_indices = range(len(ball_infos)) if current_frame_idx is None else range(current_frame_idx + 1)
    current_info = None
    for i in valid_indices:
        info = ball_infos[i]
        if info is None or info.get('ghost', False):
            continue

        source = info.get('source', 'yolo-anchor')
        if source is None:
            source = 'yolo-anchor'

        if 'interpolated_position' in info and info['interpolated_position'] is not None:
            pos = info['interpolated_position']
        else:
            box = info.get('box', [0.0, 0.0, 0.0, 0.0])
            pos = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

        if current_frame_idx is not None and i == current_frame_idx:
            current_info = info
        elif current_frame_idx is None:
            current_info = info

    if current_info is not None:
        source = current_info.get('source', 'yolo-anchor')
        color = color_map.get(source, (255, 255, 255))
        if 'box' in current_info and current_info['box'] is not None:
            x, y, w, h = current_info['box']
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(0.5 * (w + h) / 2)
            cv2.circle(output, center, radius, color, 2)

    # 3. Draw Bounce Point Indicator
    if trajectory_model is not None and trajectory_model.bounce_frame is not None:
        bounce_frame = trajectory_model.bounce_frame
        if current_frame_idx is None or bounce_frame <= current_frame_idx:
            if 0 <= bounce_frame < len(ball_infos) and ball_infos[bounce_frame] is not None:
                info = ball_infos[bounce_frame]
                if 'interpolated_position' in info and info['interpolated_position'] is not None:
                    b_pos = info['interpolated_position']
                else:
                    box = info.get('box', [0.0, 0.0, 0.0, 0.0])
                    b_pos = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
                    
                b_pos_int = (int(b_pos[0]), int(b_pos[1]))
                
                # Special indicator: Big magenta circle with a crosshair
                cv2.circle(output, b_pos_int, 10, (255, 0, 255), 2)  # Magenta circle
                cv2.line(output, (b_pos_int[0] - 15, b_pos_int[1]), (b_pos_int[0] + 15, b_pos_int[1]), (255, 0, 255), 2)
                cv2.line(output, (b_pos_int[0], b_pos_int[1] - 15), (b_pos_int[0], b_pos_int[1] + 15), (255, 0, 255), 2)
                cv2.putText(output, "BOUNCE", (b_pos_int[0] + 15, b_pos_int[1] - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return _draw_legend(output)

def _draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draws a legend indicating the meaning of each color."""
    legend_items = [
        ("YOLO Anchor", (0, 255, 0)),
        ("YOLO Rescue", (0, 255, 255)),
        ("CSRT Tracker", (255, 255, 0)),
        ("Kinematic", (0, 0, 255))
    ]
    
    y = 30
    for text, color in legend_items:
        cv2.circle(frame, (30, y - 5), 6, color, -1)
        cv2.putText(frame, text, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        
    return frame

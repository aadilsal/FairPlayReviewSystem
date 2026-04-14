import logging
import numpy as np
from typing import List, Dict, Any, Optional
from BallDetection.utils.config import POST_PROCESSOR_CONFIG
from BallDetection.pipeline.gap_classifier import classify_gaps, GapInfo
from BallDetection.pipeline.trajectory import fit_trajectory, predict_position, TrajectoryModel
from BallDetection.engines.yolo_detect import get_global_yolo_detector, yolo_detect_ball_lowconf
from BallDetection.core.validator import corridor_check
from BallDetection.engines.csrt_tracker import track_forward, track_backward, agree_and_merge
from BallDetection.core.kinematics import project_position as kinematic_project_position, find_edge_intersection
from BallDetection.core.interpolation import segment_aware_smooth
from BallDetection.utils.output import generate_output
from BallDetection.utils.ball_debug_visualizer import draw_trajectory_overlay
import json

logger = logging.getLogger(__name__)

class PostProcessor:
    def __init__(self):
        self.detector = get_global_yolo_detector()

    def process(self, ball_infos: List[Optional[Dict[str, Any]]], frames: List[np.ndarray], render_debug: bool = False, output_json_path: str = "post_processed_track.json") -> List[Dict[str, Any]]:
        """
        Main orchestration loop for the "Anchor & Rescue" pipeline.
        Processes a sequence of detections, identifies gaps, and attempts to rescue 
        missing frames using physics-informed tools.
        """
        # 1. Identify Anchors (High-confidence YOLO detections that aren't ghosts)
        # Filters out None entries and Kalman-only 'ghost' frames
        anchors = [b for b in ball_infos if b is not None and not b.get('ghost', False)]
        
        if not anchors:
            logger.warning("[POST-PROCESSOR] No anchors found in sequence. Skipping rescue pipeline.")
            return ball_infos

        first_anchor_idx = min(int(a['frame_idx']) for a in anchors if a.get('frame_idx') is not None)

        # 2. Classify Gaps (idempotent analysis of missing/ghost frame sequences)
        gaps = classify_gaps(ball_infos)
        
        # 3. Fit Trajectory Models (Splits based on bounce detection)
        trajectory_model = fit_trajectory(anchors)
        
        # 4 & 5. Run Low-Conf YOLO Rescue Pass & Corridor Filtering
        # We work on a copy of the results to avoid side-effects during iteration
        enriched_infos = list(ball_infos)
        
        rescue_conf = POST_PROCESSOR_CONFIG.get('RESCUE_CONF', 0.05)
        corridor_width = POST_PROCESSOR_CONFIG.get('CORRIDOR_WIDTH_PX', 40)

        for gap in gaps:
            if gap.end_frame < first_anchor_idx:
                continue

            for frame_idx in range(gap.start_frame, gap.end_frame + 1):
                if frame_idx < first_anchor_idx:
                    continue

                # Only attempt rescue if we have a valid trajectory model for this point in time
                pred_pos = predict_position(trajectory_model, frame_idx)
                if pred_pos is None:
                    continue
                
                # Step 5: Low-Confidence YOLO Rescue
                # We run the primary model pass but at ultra-low confidence strictly for this gap frame
                frame = frames[frame_idx]
                rescues = yolo_detect_ball_lowconf(self.detector, frame, rescue_conf)
                
                # Step 4: Corridor Filter
                # A rescue candidate is ONLY accepted if it falls within the spatial corridor of the prediction
                valid_rescues = []
                for r in rescues:
                    # r format: (x, y, w, h, conf, class_id)
                    rx, ry, rw, rh, rconf = r[:5]
                    
                    if corridor_check((rx, ry), trajectory_model, frame_idx, corridor_width):
                        # Approximate distance for logging purposes
                        dist = np.sqrt((rx - pred_pos[0])**2 + (ry - pred_pos[1])**2)
                        valid_rescues.append((r, dist))
                
                if valid_rescues:
                    # Selection: Pick highest confidence rescue that passed the corridor gate
                    (best_rescue, best_dist) = max(valid_rescues, key=lambda x: x[0][4])
                    rx, ry, rw, rh, rconf = best_rescue[:5]
                    
                    enriched_infos[frame_idx] = {
                        'box': [float(rx), float(ry), float(rw), float(rh)],
                        'conf': float(rconf),
                        'source': 'yolo-rescue',
                        'frame_idx': frame_idx,
                        'interpolated_position': (float(rx), float(ry)),
                        'state': 2, # Force TRACKING state representation
                        'miss_streak': 0
                    }
                    logger.info(f"[POST-PROCESSOR] Rescued frame {frame_idx} via YOLO (dist={best_dist:.1f}px, conf={rconf:.2f})")

        # 6. Run CSRT tracking (Phase 4 — stubbed)
        self._run_csrt_rescue(enriched_infos, frames, gaps, trajectory_model, first_anchor_idx)

        # 7. Run kinematic fallback (Phase 5 — stubbed)
        self._run_kinematic_fallback(enriched_infos, trajectory_model, first_anchor_idx)

        # 8. Final Segment-Aware Kalman Smoothing (Phase 5)
        self._run_final_smoothing(enriched_infos, trajectory_model, first_anchor_idx)

        # 9. Output Generation
        final_output = generate_output(enriched_infos)
        
        if output_json_path:
            try:
                with open(output_json_path, 'w') as f:
                    json.dump(final_output, f, indent=4)
                logger.info(f"[POST-PROCESSOR] Saved final annotations to {output_json_path}")
            except Exception as e:
                logger.error(f"[POST-PROCESSOR] Failed to save output JSON: {str(e)}")

        # 10. Draw Debug Visualization
        if render_debug:
            logger.info("[POST-PROCESSOR] Drawing debug trajectory overlay on frames...")
            for i in range(len(frames)):
                # Modify frame in-place so caller can just render back the list of frames
                frames[i] = draw_trajectory_overlay(frames[i], enriched_infos, trajectory_model, current_frame_idx=i)

        return enriched_infos

    def _run_csrt_rescue(self, ball_infos, frames, gaps, model, first_anchor_idx: int):
        """Phase 4: Bidirectional CSRT tracking with agreement verification."""
        for gap in gaps:
            if gap.end_frame < first_anchor_idx:
                continue

            # We only use CSRT if there are still None/ghost frames in the gap
            # because YOLO rescue might have already filled some or all of it.
            # Find the longest contiguous sub-gap of unrescued frames
            need_csrt = [i for i in range(gap.start_frame, gap.end_frame + 1) 
                         if i >= first_anchor_idx and (ball_infos[i] is None or ball_infos[i].get('ghost', False))]
            
            if not need_csrt:
                continue
                
            is_occlusion = (gap.gap_type == 'occlusion')
            
            # Find closest valid anchors specifically for CSRT initialization
            start_bbox = gap.pre_anchor['box'] if gap.pre_anchor else None
            end_bbox = gap.post_anchor['box'] if gap.post_anchor else None
            
            f_res, b_res = {}, {}
            
            if start_bbox:
                f_res = track_forward(frames, gap.start_frame - 1, start_bbox, gap.end_frame, is_occlusion)
                
            if end_bbox:
                b_res = track_backward(frames, gap.end_frame + 1, end_bbox, gap.start_frame, is_occlusion)
                
            # Agreement Merge
            if f_res or b_res:
                merged = agree_and_merge(f_res, b_res, gap.gap_type)
                
                for f_idx, result in merged.items():
                    # Only apply if it's still missing (don't overwrite YOLO rescues)
                    if f_idx >= first_anchor_idx and (ball_infos[f_idx] is None or ball_infos[f_idx].get('ghost', False)):
                        box = result['box']
                        ball_infos[f_idx] = {
                            'box': box,
                            'interpolated_position': result['interpolated_position'],
                            'source': result['source'],
                            'frame_idx': f_idx,
                            'conf': 0.0, # CSRT doesn't give YOLO-like confidence
                            'state': 2,
                            'miss_streak': 0
                        }
                        # If edge suspected, preserve the divergent boxes for Phase 5 kinematics
                        if result['source'] == 'edge-suspected':
                            ball_infos[f_idx]['f_box'] = result['f_box']
                            ball_infos[f_idx]['b_box'] = result['b_box']
                        
                        logger.info(f"[POST-PROCESSOR] Rescued frame {f_idx} via CSRT ({result['source']})")

    def _run_kinematic_fallback(self, ball_infos, model, first_anchor_idx: int):
        """Phase 5: Pure physics-only projection for total blackouts and edge intersection."""
        # 1. Resolve edge-suspected frames (Trackers disagreed during occlusion)
        for i in range(first_anchor_idx, len(ball_infos)):
            info = ball_infos[i]
            if info is not None and info.get('source') == 'edge-suspected':
                # Build forward arc (recent past + current tracker prediction)
                f_arc = []
                for j in range(max(0, i-5), i):
                    prev = ball_infos[j]
                    if prev and not prev.get('ghost', False) and prev.get('source') in ['yolo', 'yolo-rescue', 'csrt-forward', 'csrt-agreed']:
                        f_arc.append(prev['interpolated_position'])
                
                if 'f_box' in info:
                    f_box = info['f_box']
                    f_arc.append((f_box[0] + f_box[2]/2, f_box[1] + f_box[3]/2))

                # Build backward arc (current tracker prediction + near future)
                b_arc = []
                if 'b_box' in info:
                    b_box = info['b_box']
                    b_arc.append((b_box[0] + b_box[2]/2, b_box[1] + b_box[3]/2))
                
                for j in range(i+1, min(len(ball_infos), i+6)):
                    nxt = ball_infos[j]
                    if nxt and not nxt.get('ghost', False) and nxt.get('source') in ['yolo', 'yolo-rescue', 'csrt-backward', 'csrt-agreed']:
                        b_arc.append(nxt['interpolated_position'])

                # Find precise sub-frame intersection 
                intersect = find_edge_intersection(f_arc, b_arc)
                if intersect:
                    info['interpolated_position'] = intersect
                    info['box'][0] = intersect[0] - info['box'][2]/2
                    info['box'][1] = intersect[1] - info['box'][3]/2
                    logger.info(f"[POST-PROCESSOR] Edge visually resolved at frame {i}: {intersect}")

        # 2. Fill in remaining None/ghost frames with pure projection
        for i in range(first_anchor_idx, len(ball_infos)):
            info = ball_infos[i]
            if info is None or info.get('ghost', False):
                pos = kinematic_project_position(model, i)
                if pos:
                    # Nominal 20x20 bounding box for kinematic fallbacks
                    box = [float(pos[0]) - 10.0, float(pos[1]) - 10.0, 20.0, 20.0]
                    ball_infos[i] = {
                        'box': box,
                        'interpolated_position': pos,
                        'source': 'kinematic',
                        'frame_idx': i,
                        'conf': 0.0,
                        'state': 2,
                        'miss_streak': 0
                    }
                    logger.info(f"[POST-PROCESSOR] Rescued frame {i} via Kinematics")

    def _run_final_smoothing(self, ball_infos, model, first_anchor_idx: int):
        """Phase 5: Re-running Kalman filter while preserving the 'V' at bounce points."""
        smoothed_positions = segment_aware_smooth(ball_infos, model.bounce_frame)
        
        for i, pos in enumerate(smoothed_positions):
            if i >= first_anchor_idx and ball_infos[i] is not None:
                # Update the interpolated position with the segmented smoothing result
                ball_infos[i]['interpolated_position'] = pos

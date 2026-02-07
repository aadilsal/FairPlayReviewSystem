# file: detection_pipeline.py
import cv2
import os
import json

# Logic Modules
# from exp_ball_detector import detect_ball_on_frame as detect_ball_on_frame
from BallDetection.ball_detector import detect_ball_on_frame
from BatsmanDetection.pose_estimator import estimate_pose
from BatsmanDetection.person_detector import detect_persons
from BatsmanDetection.bat_detector import detect_bat
from BatsmanDetection.Batsman_finder import BatsmanFinder
from BatsmanDetection.Batsman_tracker import BatsmanTracker
from WicketDetection.wicket_detector import detect_wicket, WicketLineEstimator

from utils.visualizer import visualize_frame
from BallDetection.pitch_plane import PitchPlaneEstimator
from utils.lbw_decision import compute_lbw_decision, DECISION_NO_DECISION


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def process_frames_pipeline(
    frame_paths,
    person_conf=0.5,
    bat_conf=0.3,
    pad_conf=0.3,
    iou_thresh=0.05,
    consec_required=3,
    wicket_conf=0.25,
    display=True
):
    # Initialize Batsman Logic
    batsman_finder = BatsmanFinder(
        iou_thresh=iou_thresh,
        consec_required=consec_required
    )
    batsman_tracker = BatsmanTracker()
    pitch_estimator = PitchPlaneEstimator(warmup_frames=10)
    pitch_model = None
    wicket_line_estimator = WicketLineEstimator(warmup_frames=12, min_samples=6)
    wicket_line = None
    
    tracking_active = False

    # Calculate total frames for progress tracking
    total_frames = len(frame_paths)
    print(f"[INFO] Starting pipeline on {total_frames} frames...")

    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            current_progress = frame_idx + 1
            print(f"[INFO] Processing Frame {current_progress}/{total_frames} : {os.path.basename(frame_path)}")

            # 1. READ CLEAN FRAME
            clean_frame = cv2.imread(frame_path)
            if clean_frame is None:
                print(f"[WARN] Could not read {frame_path}")
                continue

            if pitch_estimator and not pitch_estimator.is_ready() and not pitch_estimator.is_failed():
                pitch_estimator.add_frame(clean_frame)
                if pitch_estimator.is_ready():
                    pitch_model = pitch_estimator.get_model()
            elif pitch_estimator and pitch_estimator.is_ready():
                pitch_model = pitch_estimator.get_model()

            metadata = {
                "frame_index": frame_idx,
                "tracking_active": tracking_active,
                "detections": []
            }
            if pitch_model:
                metadata["pitch_model"] = {
                    "left_line": pitch_model.get("left_line"),
                    "right_line": pitch_model.get("right_line"),
                    "top_y": pitch_model.get("top_y"),
                    "bottom_y": pitch_model.get("bottom_y"),
                    "polygon": pitch_model.get("polygon"),
                    "confidence": pitch_model.get("confidence")
                }
            if wicket_line:
                metadata["wicket_line"] = {
                    "x": wicket_line.get("x"),
                    "y_top": wicket_line.get("y_top"),
                    "y_bottom": wicket_line.get("y_bottom")
                }

            # Temp variables for visualization
            det_ball = None
            det_persons = []
            det_batsman_box = None
            det_wickets = []
            det_pose = []
            det_bats = []  

            # ======================================================
            # 2️⃣ GATHER DATA (Run Detectors on Clean Copies)
            # ======================================================

            # A. Ball Detection
            _, det_ball = detect_ball_on_frame(frame_idx=frame_idx, frame=clean_frame.copy(), batsman_box=det_batsman_box, wicket_line=wicket_line)
            lbw_status = None
            lbw_decision = None
            lbw_reason = None
            if det_ball:
                metadata["detections"].append({"label": "Ball", "data": det_ball})
                state = det_ball.get("state") if isinstance(det_ball, dict) else None
                if state:
                    impact_point = state.get("impact_point")
                    would_hit_stumps = state.get("would_hit_stumps")
                    confidence = state.get("confidence")
                    if pitch_model is None or impact_point is None or would_hit_stumps is None:
                        lbw_status = "INSUFFICIENT DATA"
                        lbw_decision = DECISION_NO_DECISION
                        lbw_reason = "insufficient_data"
                    elif confidence is None or confidence < 0.55:
                        lbw_status = "LOW CONFIDENCE"
                        lbw_decision = DECISION_NO_DECISION
                        lbw_reason = "low_confidence"
                    else:
                        lbw_decision, lbw_reason = compute_lbw_decision(
                            impact_point=impact_point,
                            pitch_model=pitch_model,
                            would_hit_stumps=would_hit_stumps,
                            confidence=confidence
                        )
            else:
                lbw_status = "TRACK LOST"
                lbw_decision = DECISION_NO_DECISION
                lbw_reason = "track_lost"

            if lbw_decision:
                metadata["lbw_decision"] = {
                    "decision": lbw_decision,
                    "reason": lbw_reason,
                    "status": lbw_status,
                    "impact_point": state.get("impact_point") if det_ball and state else None,
                    "would_hit_stumps": state.get("would_hit_stumps") if det_ball and state else None,
                    "confidence": state.get("confidence") if det_ball and state else None
                }

            # B. Wicket Detection
            _, det_wickets = detect_wicket(clean_frame.copy(), conf=wicket_conf)
            if det_wickets:
                metadata["detections"].extend(det_wickets)
                if wicket_line is None:
                    wicket_line_estimator.add_detections(det_wickets)
                    if wicket_line_estimator.is_ready():
                        wicket_line = wicket_line_estimator.get_model()
            
            # C. Bat Detection
            _, det_bats = detect_bat(clean_frame.copy(), conf=bat_conf)
            for b in det_bats:
                 metadata["detections"].append({"label": "Bat", "box": b["box"], "conf": b.get("conf", 0.0)})

            # D. Batsman Logic
            if not tracking_active:
                # E. General Person Detection
                _, det_persons = detect_persons(clean_frame.copy(), person_conf=person_conf)
                for p in det_persons:
                    metadata["detections"].append({"label": "Person", "box": list(p[:4]), "conf": 0.0})

                # Search Mode: Pass 'det_persons' AND 'det_bats'
                _, finder_meta = batsman_finder.process_frame(
                    clean_frame.copy(), 
                    det_persons, 
                    det_bats,    
                    frame_idx
                )
                
                if finder_meta.get("batsman_confirmed", False):
                    bbox = finder_meta["batsman_bbox"]
                    if batsman_tracker.init_tracker(clean_frame, bbox):
                        tracking_active = True
                        metadata["tracking_active"] = True
                        det_batsman_box = list(map(int, bbox))
                        print(f"[INFO] ✅ Batsman confirmed at frame {frame_idx}")
            else:
                # Track Mode
                ok, bbox = batsman_tracker.update(clean_frame.copy())
                if ok:
                    det_batsman_box = list(map(int, bbox))
                    metadata["tracking_active"] = True
                    metadata["detections"].append({"label": "Batsman", "box": det_batsman_box, "tracked": True})
                else:
                    print(f"[WARN] Tracker lost at frame {frame_idx}")
                    tracking_active = False
                    # Re-init finder
                    batsman_finder = BatsmanFinder(iou_thresh=iou_thresh, consec_required=consec_required)
                    batsman_tracker = BatsmanTracker()


            # F. Pose Estimation
            if det_batsman_box: 
                # Case 1: Batsman is confirmed. Run ONLY on the batsman.
                _, kps = estimate_pose(clean_frame.copy(), bbox=det_batsman_box)
                det_pose.extend(kps)
            else:
                # Case 2: No batsman. Run on all valid 'Persons'.
                for p in det_persons:
                    box = list(p[:4]) 
                    _, kps = estimate_pose(clean_frame.copy(), bbox=box)
                    det_pose.extend(kps)

            # G. Pad Detection 
            det_pads = []
            # _, det_pads = detect_pads(clean_frame.copy(), det_pose, conf=pad_conf)
            # for p in det_pads:
            #     metadata["detections"].append({"label": "Pad", "box": p["box"], "conf": p.get("conf", 0.0)})

            # ======================================================
            # 3️⃣ VISUALIZATION & OUTPUT
            # ======================================================
            
            # Draw everything on a fresh copy
            pitch_status = None
            if pitch_estimator:
                if pitch_estimator.is_ready():
                    pitch_status = "READY"
                elif pitch_estimator.is_failed():
                    pitch_status = "FAILED"
                else:
                    pitch_status = "WARMUP"

            vis_frame = visualize_frame(
                clean_frame.copy(),
                det_ball, 
                det_persons, 
                det_batsman_box, 
                det_wickets, 
                det_bats,
                det_pads, 
                det_pose,
                frame_idx,
                pitch_model=pitch_model,
                pitch_status=pitch_status,
                wicket_line=wicket_line,
                lbw_decision=lbw_decision,
                lbw_status=lbw_status
            )

            # --- ADDING FRAME COUNTER ---
            cv2.putText(
                vis_frame, 
                f"Frame: {current_progress}/{total_frames}", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 255), 
                2
            )

            # Save Image
            cv2.imwrite(frame_path, vis_frame)
            
            # Save Metadata
            meta_path = os.path.splitext(frame_path)[0] + ".json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Display
            if display:
                cv2.imshow("FairPlayReviewSystem", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Pipeline interrupted by user.")
                    break
    
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the pipeline: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("[INFO] Cleaning up resources...")
        cv2.destroyAllWindows()
        print("[INFO] Pipeline processing finished.")
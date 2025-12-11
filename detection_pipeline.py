# file: detection_pipeline.py
import cv2
import os
import json
import numpy as np

# Logic Modules
from ball_detector import detect_ball_on_frame
from pose_estimator import estimate_pose
from person_detector import detect_persons
from Batsman_finder import BatsmanFinder
from Batsman_tracker import BatsmanTracker
from visualizer import visualize_frame

from wicket_detector import detect_wicket
init_wicket_tracker = None

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def process_frames_pipeline(
    frame_paths,
    person_conf=0.5,
    bat_conf=0.3,
    iou_thresh=0.12,
    consec_required=3,
    wicket_conf=0.25,
    display=True
):
    # Initialize Batsman Logic
    batsman_finder = BatsmanFinder(
        person_conf=person_conf,
        bat_conf=bat_conf,
        iou_thresh=iou_thresh,
        consec_required=consec_required
    )
    batsman_tracker = BatsmanTracker()
    
    tracking_active = False

    for frame_idx, frame_path in enumerate(frame_paths):

        # 1. READ CLEAN FRAME
        clean_frame = cv2.imread(frame_path)
        if clean_frame is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        metadata = {
            "frame_index": frame_idx,
            "tracking_active": tracking_active,
            "detections": []
        }

        # Temp variables for visualization
        det_ball = None
        det_persons = []
        det_batsman_box = None
        det_wickets = []
        det_pose = []

        # ======================================================
        # 2️⃣ GATHER DATA (Run Detectors on Clean Copies)
        # ======================================================

        # A. General Person Detection
        _, det_persons = detect_persons(clean_frame.copy(), person_conf=person_conf)
        for p in det_persons:
             metadata["detections"].append({"label": "Person", "box": list(p[:4]), "conf": 0.0})

        # B. Ball Detection
        _, det_ball = detect_ball_on_frame(clean_frame.copy())
        if det_ball:
            metadata["detections"].append({"label": "Ball", "data": det_ball})

        # C. Batsman Logic
        if not tracking_active:
            # Search Mode
            _, finder_meta = batsman_finder.process_frame(clean_frame.copy(), frame_idx)
            
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
                batsman_finder = BatsmanFinder(person_conf, bat_conf, iou_thresh, consec_required)
                batsman_tracker = BatsmanTracker()

        # D. Wicket Detection
        _, det_wickets = detect_wicket(clean_frame.copy(), conf=wicket_conf)
        if det_wickets:
            metadata["detections"].extend(det_wickets)

        # E. Pose Estimation
        pose_crop_box = det_batsman_box if det_batsman_box else None
        _, det_pose = estimate_pose(clean_frame.copy(), bbox=pose_crop_box)

        # ======================================================
        # 3️⃣ VISUALIZATION & OUTPUT
        # ======================================================
        
        # Draw everything on a fresh copy
        vis_frame = visualize_frame(
            clean_frame.copy(),
            det_ball, 
            det_persons, 
            det_batsman_box, 
            det_wickets, 
            det_pose, 
            frame_idx  # <--- Now passing frame_idx
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
                break

    cv2.destroyAllWindows()
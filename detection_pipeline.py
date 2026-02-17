import cv2
import os
import json
import numpy as np

from ball_detector import detect_ball
from pose_estimator import estimate_pose
from person_detector import detect_persons
from pad_detector import detect_pads
from bat_detector import detect_bat 
from Batsman_finder import BatsmanFinder
from Batsman_tracker import BatsmanTracker
from wicket_detector import detect_wicket
from visualizer import visualize_frame
from global_config import GLOBAL_CONFIG
from preprocessing import preprocess_frame

def process_frames_pipeline(
    frame_paths,
    person_conf,
    bat_conf,
    pad_conf,
    iou_thresh,
    consec_required,
    wicket_conf,
    preprocess,
    display
):
    batsman_finder = BatsmanFinder(
        iou_thresh=iou_thresh,
        consec_required=consec_required
    )
    batsman_tracker = BatsmanTracker()
    tracking_active = False

    for frame_idx, frame_path in enumerate(frame_paths):
        clean_frame = cv2.imread(frame_path)
        if clean_frame is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        if preprocess:
            frame = preprocess_frame(
                clean_frame,
                color_mode=None,
                enable_deblur=GLOBAL_CONFIG['enable_deblur'],
                enable_sharpen=GLOBAL_CONFIG['enable_sharpen'],
                enable_clahe=GLOBAL_CONFIG['enable_clahe'],
                blur_threshold=GLOBAL_CONFIG['blur_threshold']
            )
        else:
            frame = clean_frame.copy()

        metadata = {
            "frame_index": frame_idx,
            "tracking_active": tracking_active,
            "detections": []
        }

        det_ball = None
        det_persons = []
        det_batsman_box = None
        det_wickets = []
        det_pose = []
        det_bats = []  
        det_pads = []

        # A. Ball Detection
        det_ball = detect_ball(frame=frame.copy(), frame_idx=frame_idx)
        if det_ball:
            metadata["detections"].append({"label": "Ball", "data": det_ball})

        # # B. Wicket Detection
        # _, det_wickets = detect_wicket(frame.copy(), conf=wicket_conf)
        # if det_wickets:
        #     metadata["detections"].extend(det_wickets)
        
        # # C. Bat Detection
        # _, det_bats = detect_bat(frame.copy(), conf=bat_conf)
        # for b in det_bats:
        #      metadata["detections"].append({"label": "Bat", "box": b["box"], "conf": b.get("conf", 0.0)})

        # # D. Batsman Logic
        # if not tracking_active:
        #     # E. General Person Detection
        #     _, det_persons = detect_persons(frame.copy(), person_conf=person_conf)
        #     for p in det_persons:
        #         metadata["detections"].append({"label": "Person", "box": list(p[:4]), "conf": 0.0})

        #     # Search Mode: Pass 'det_persons' AND 'det_bats'
        #     _, finder_meta = batsman_finder.process_frame(
        #         frame.copy(), 
        #         det_persons, 
        #         det_bats,    
        #         frame_idx
        #     )
            
        #     if finder_meta.get("batsman_confirmed", False):
        #         bbox = finder_meta["batsman_bbox"]
        #         if batsman_tracker.init_tracker(frame, bbox):
        #             tracking_active = True
        #             metadata["tracking_active"] = True
        #             det_batsman_box = list(map(int, bbox))
        #             print(f"[INFO] ✅ Batsman confirmed at frame {frame_idx}")
        # else:
        #     # Track Mode
        #     ok, bbox = batsman_tracker.update(frame.copy())
        #     if ok:
        #         det_batsman_box = list(map(int, bbox))
        #         metadata["tracking_active"] = True
        #         metadata["detections"].append({"label": "Batsman", "box": det_batsman_box, "tracked": True})
        #     else:
        #         print(f"[WARN] Tracker lost at frame {frame_idx}")
        #         tracking_active = False
        #         batsman_finder = BatsmanFinder(iou_thresh=iou_thresh, consec_required=consec_required)
        #         batsman_tracker = BatsmanTracker()

        # # F. Pose Estimation
        # if det_batsman_box: 
        #     _, kps = estimate_pose(frame.copy(), bbox=det_batsman_box)
        #     det_pose.extend(kps)
        # else:
        #     for p in det_persons:
        #         box = list(p[:4]) 
        #         _, kps = estimate_pose(frame.copy(), bbox=box)
        #         det_pose.extend(kps)

        # # G. Pad Detection 
        # # _, det_pads = detect_pads(frame.copy(), det_pose, conf=pad_conf)
        # # for p in det_pads:
        # #     metadata["detections"].append({"label": "Pad", "box": p["box"], "conf": p.get("conf", 0.0)})

        vis_frame = visualize_frame(
            frame.copy(),
            det_ball, 
            det_persons, 
            det_batsman_box, 
            det_wickets, 
            det_bats,
            det_pads, 
            det_pose,
            frame_idx
        )

        cv2.imwrite(frame_path, vis_frame)
        meta_path = os.path.splitext(frame_path)[0] + ".json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        if display:
            cv2.imshow("FairPlayReviewSystem", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
# file: detection_pipeline.py
import cv2
import os
import json

from ball_detector import detect_ball_on_frame
from detection.pose_detector import estimate_pose
from Batsman_finder import BatsmanFinder
from Batsman_tracker import BatsmanTracker


# ---------------------------------------------------------
# HUD / STATUS OVERLAY
# ---------------------------------------------------------
def draw_status_overlay(
    frame,
    num_persons,
    num_bats,
    best_iou,
    consec_count,
    consec_required,
    mode
):
    if mode=="TRACK":
        text = (
        f"Matches: {consec_count}/{consec_required} | "
        f"Mode: {mode}"
    )
    else:
         text = (
        f"Persons: {num_persons} | "
        f"Bats: {num_bats} | "
        f"Best IoU: {best_iou:.3f} | "
        f"Matches: {consec_count}/{consec_required} | "
        f"Mode: {mode}"
    )   
    

    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def process_frames_pipeline(
    frame_paths,
    person_conf=0.5,
    bat_conf=0.3,
    iou_thresh=0.12,
    consec_required=3,
    display=True
):

    batsman_finder = BatsmanFinder(
        person_conf=person_conf,
        bat_conf=bat_conf,
        iou_thresh=iou_thresh,
        consec_required=consec_required
    )

    batsman_tracker = BatsmanTracker()
    tracking_active = False
    last_finder_meta = None

    for frame_idx, frame_path in enumerate(frame_paths):

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        metadata = {
            "frame_index": frame_idx,
            "tracking_active": tracking_active,
            "detections": []
        }

        # ======================================================
        # 1️⃣ BALL DETECTION (ALWAYS ON)
        # ======================================================
        frame, ball_info = detect_ball_on_frame(frame, frame_idx=frame_idx)
        if ball_info:
            # Keep full ball_info for diagnostics
            metadata["detections"].append({
                "label": "Ball",
                "data": ball_info
            })

            # Also add a box entry similar to how batsman is stored
            try:
                bx1, by1, bx2, by2 = ball_info.get('bbox', [0, 0, 0, 0])
                bw = int(bx2 - bx1)
                bh = int(by2 - by1)
                metadata["detections"].append({
                    "label": "Ball",
                    "box": [int(bx1), int(by1), bw, bh],
                    "confidence": float(ball_info.get('confidence', 0.0)),
                    "tracked": False
                })
            except Exception:
                # ignore if malformed ball_info
                pass

            # Also append to global detections.json (matching existing format)
            try:
                detections_path = os.path.join(os.getcwd(), "detections.json")
                if os.path.exists(detections_path):
                    with open(detections_path, "r", encoding="utf-8") as df:
                        existing = json.load(df)
                        if not isinstance(existing, list):
                            existing = []
                else:
                    existing = []

                x1, y1, x2, y2 = ball_info.get('bbox', [0, 0, 0, 0])
                conf = ball_info.get('confidence', 0.0)
                entry = {
                    "frame_index": frame_idx,
                    "frame_id": os.path.basename(frame_path),
                    "x_min": float(x1),
                    "y_min": float(y1),
                    "x_max": float(x2),
                    "y_max": float(y2),
                    "confidence": float(conf),
                    "class_id": None,
                    "class_name": "sports ball"
                }

                existing.append(entry)
                with open(detections_path, "w", encoding="utf-8") as df:
                    json.dump(existing, df, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to append ball detection to detections.json: {e}")

        # ======================================================
        # 2️⃣ BATSMAN FINDING / TRACKING
        # ======================================================
        if not tracking_active and batsman_finder.state != BatsmanFinder.CONFIRMED:

            frame, finder_meta = batsman_finder.process_frame(frame, frame_idx)
            last_finder_meta = finder_meta

            print(
                f"[DBG] frame={frame_idx} "
                f"persons={finder_meta.get('num_persons', 0)} "
                f"bats={finder_meta.get('num_bats', 0)} "
                f"best_iou={finder_meta.get('best_iou', 0.0):.3f} "
                f"consec={finder_meta.get('consec_count', 0)} "
                f"state={finder_meta.get('state', 'NA')}"
            )

            draw_status_overlay(
                frame,
                finder_meta["num_persons"],
                finder_meta["num_bats"],
                finder_meta["best_iou"],
                finder_meta["consec_count"],
                consec_required,
                finder_meta["state"]
            )

            if finder_meta.get("batsman_confirmed", False):
                bbox = finder_meta["batsman_bbox"]
                ok = batsman_tracker.init_tracker(frame, bbox)

                if ok:
                    tracking_active = True
                    metadata["tracking_active"] = True
                    print(f"[INFO] ✅ Batsman confirmed at frame {frame_idx}")
                else:
                    print("[WARN] Tracker initialization failed")

        else:
            ok, bbox = batsman_tracker.update(frame)
            metadata["tracking_active"] = True

            if ok:
                x, y, w, h = map(int, bbox)
                
                metadata["detections"].append({"label":"Batsman",
                                               "box":[x,y,w,h],
                                               "tracked":True})
                    
                cv2.rectangle(frame, 
                              (x, y), 
                              (x + w, y + h), 
                              (255, 0, 0), 2)

                cv2.putText(
                    frame,
                    "Batsman (Tracked)",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

                draw_status_overlay(
                    frame,
                    0,
                    0,
                    0.00,
                    #last_finder_meta["num_persons"],
                    #last_finder_meta["num_bats"],
                    #last_finder_meta["best_iou"],
                    consec_required,
                    consec_required,
                    "TRACK"
                )

            else:
                print(f"[WARN] Tracker lost at frame {frame_idx}")
                tracking_active = False
                batsman_finder = BatsmanFinder(
                    person_conf=person_conf,
                    bat_conf=bat_conf,
                    iou_thresh=iou_thresh,
                    consec_required=consec_required
                )
                batsman_tracker = BatsmanTracker()
                last_finder_meta = None

        # ======================================================
        # 3️⃣ POSE ESTIMATION
        # ======================================================
        frame, pose_data = estimate_pose(frame)

        # ======================================================
        # 4️⃣ SAVE
        # ======================================================
        cv2.imwrite(frame_path, frame)

        meta_path = os.path.splitext(frame_path)[0] + ".json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # ======================================================
        # 5️⃣ DISPLAY
        # ======================================================
        if display:
            cv2.imshow("FairPlayReviewSystem", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

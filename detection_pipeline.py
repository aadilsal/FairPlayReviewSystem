import cv2
import json
import os
import re
import os
import re
import numpy as np
import logging
import logging

from BallDetection.pipeline.ball_detector import detect_ball
from BallDetection.pipeline.trajectory import fit_trajectory
from BallDetection.pipeline.trajectory import fit_trajectory
from pose_estimator import estimate_pose
from person_detector import detect_persons
from pad_detector import detect_pads
from bat_detector import detect_bat
from bat_detector import detect_bat
from Batsman_finder import BatsmanFinder
from Batsman_tracker import BatsmanTracker
from wicket_detector import detect_wicket
from visualizer import visualize_frame
from global_config import GLOBAL_CONFIG
from preprocessing import preprocess_frame
from LbwDecision.lbw_analyzer import (
    analyze_lbw_sequence,
    build_anchors_from_ball_infos,
    lbw_overlay_for_api,
)
from LbwDecision.lbw_review_card import render_lbw_review_card

logger = logging.getLogger("fairplay.pipeline")

def _safe_video_stem(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    return (s[:120] or "video").strip("_")

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def _pick_pose_for_batsman(keypoints_list, batsman_box):
    """Return only the pose instance that best overlaps the batsman bbox."""
    if not keypoints_list or not batsman_box or len(batsman_box) < 4:
        return []

    bx, by, bw, bh = [float(v) for v in batsman_box[:4]]
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh
    b_cx = (b_x1 + b_x2) / 2.0
    b_cy = (b_y1 + b_y2) / 2.0

    def _pose_bounds(kps):
        arr = np.asarray(kps)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        valid = (arr[:, 0] > 0) & (arr[:, 1] > 0)
        if not np.any(valid):
            return None
        pts = arr[valid, :2]
        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
        return float(x1), float(y1), float(x2), float(y2)

    best = None
    best_score = None
    for kps in keypoints_list:
        pb = _pose_bounds(kps)
        if pb is None:
            continue
        p_x1, p_y1, p_x2, p_y2 = pb

        inter_x1 = max(b_x1, p_x1)
        inter_y1 = max(b_y1, p_y1)
        inter_x2 = min(b_x2, p_x2)
        inter_y2 = min(b_y2, p_y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        p_area = max(1.0, (p_x2 - p_x1) * (p_y2 - p_y1))
        b_area = max(1.0, (b_x2 - b_x1) * (b_y2 - b_y1))
        iou = inter / (p_area + b_area - inter + 1e-9)

        p_cx = (p_x1 + p_x2) / 2.0
        p_cy = (p_y1 + p_y2) / 2.0
        center_dist = float(np.hypot(p_cx - b_cx, p_cy - b_cy))

        # Higher IoU is better, then closer center.
        score = (iou, -center_dist)
        if best_score is None or score > best_score:
            best_score = score
            best = kps

    return [best] if best is not None else []


def process_frames_pipeline(
    frame_paths,
    person_conf,
    bat_conf,
    pad_conf,
    iou_thresh,
    consec_required,
    wicket_conf,
    preprocess,
    display,
    wicket_override=None,
    dynamic_wicket_detection=True,
    video_stem=None,
):
    batsman_finder = BatsmanFinder(
        iou_thresh=iou_thresh,
        consec_required=consec_required
    )
    batsman_tracker = BatsmanTracker()
    tracking_active = False

    all_ball_infos = []
    detection_frames = []
    frame_records = []
    n_paths = len(frame_paths)
    if n_paths == 0:
        logger.warning("No frames provided to process_frames_pipeline")
        return

    progress_step = max(1, n_paths // 10)
    logger.info("Pipeline pass 1/2 started: analyzing %s frames", n_paths)

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx % progress_step == 0:
            logger.info("Pipeline pass 1/2 progress: %s/%s frames", frame_idx, n_paths)
        clean_frame = cv2.imread(frame_path)
        if clean_frame is None:
            print(f"[WARN] Could not read {frame_path}")
            all_ball_infos.append(None)
            all_ball_infos.append(None)
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

        detection_frames.append(frame.copy())

        metadata = {
            "frame_index": frame_idx,
            "tracking_active": tracking_active,
            "detections": []
        }

        det_ball = detect_ball(frame=frame.copy(), frame_idx=frame_idx)
        all_ball_infos.append(det_ball)
        if det_ball:
            metadata["detections"].append({"label": "Ball", "data": det_ball})

        det_persons = []
        det_batsman_box = None
        det_wickets = []
        det_pose = []
        det_bats = []
        det_pads = []

        if wicket_override is not None:
            det_wickets = wicket_override
            if det_wickets:
                metadata["detections"].extend(det_wickets)
        elif dynamic_wicket_detection:
            _, det_wickets = detect_wicket(frame.copy(), conf=wicket_conf)
            if det_wickets:
                metadata["detections"].extend(det_wickets)
        else:
            det_wickets = []

        _, det_bats = detect_bat(frame.copy(), conf=bat_conf)
        for b in det_bats:
            metadata["detections"].append({"label": "Bat", "box": b["box"], "conf": b.get("conf", 0.0)})

        if not tracking_active:
            _, det_persons = detect_persons(frame.copy(), person_conf=person_conf)
            for p in det_persons:
                metadata["detections"].append({"label": "Person", "box": list(p[:4]), "conf": 0.0})

            _, finder_meta = batsman_finder.process_frame(
                frame.copy(),
                det_persons,
                det_bats,
                frame_idx
            )


            if finder_meta.get("batsman_confirmed", False):
                bbox = finder_meta["batsman_bbox"]
                if batsman_tracker.init_tracker(frame, bbox):
                    tracking_active = True
                    metadata["tracking_active"] = True
                    det_batsman_box = list(map(int, bbox))
                    logger.info("Batsman confirmed at frame %s", frame_idx)
                    logger.info("Batsman confirmed at frame %s", frame_idx)
        else:
            ok, bbox = batsman_tracker.update(frame.copy())
            if ok:
                det_batsman_box = list(map(int, bbox))
                metadata["tracking_active"] = True
                metadata["detections"].append({"label": "Batsman", "box": det_batsman_box, "tracked": True})
            else:
                logger.warning("Tracker lost at frame %s", frame_idx)
                logger.warning("Tracker lost at frame %s", frame_idx)
                tracking_active = False
                batsman_finder = BatsmanFinder(iou_thresh=iou_thresh, consec_required=consec_required)
                batsman_tracker = BatsmanTracker()

        if det_batsman_box:
            _, kps = estimate_pose(frame.copy(), bbox=det_batsman_box)
            # Keep only the pose that best corresponds to the batsman box.
            det_pose = _pick_pose_for_batsman(kps, det_batsman_box)
            _, det_pads = detect_pads(frame.copy(), det_pose, conf=pad_conf)
        else:
            # Do not run pose/pad estimation for non-batsman persons.
            det_pose = []
            det_pads = []

        for p in det_pads:
            metadata["detections"].append({"label": "Pad", "box": p["box"], "conf": p.get("conf", 0.0)})

        frame_records.append({
            "path": frame_path,
            "det_persons": det_persons,
            "det_batsman_box": det_batsman_box,
            "det_wickets": det_wickets,
            "det_bats": det_bats,
            "det_pads": det_pads,
            "det_pose": det_pose,
            "metadata": metadata,
        })

        if display:
            live_frame = visualize_frame(
                frame.copy(),
                all_ball_infos,
                det_persons,
                det_batsman_box,
                det_wickets,
                det_bats,
                det_pads,
                det_pose,
                frame_idx,
                lbw_overlay=None,
            )
            cv2.putText(
                live_frame,
                "LIVE: detection / tracking",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("FairPlayReviewSystem - Live", live_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                display = False

    for rec, ball_info in zip(frame_records, all_ball_infos):
        detections = [d for d in rec["metadata"]["detections"] if d.get("label") != "Ball"]
        if ball_info is not None and not ball_info.get("ghost", False):
            detections.insert(0, {"label": "Ball", "data": ball_info})
        rec["metadata"]["detections"] = detections

    anchors = build_anchors_from_ball_infos(all_ball_infos)
    trajectory_model = fit_trajectory(anchors)

    wickets_by_fi = {}
    pads_by_fi = {}
    batsman_by_fi = {}
    bats_by_fi = {}
    for r in frame_records:
        fi = int(r["metadata"]["frame_index"])
        wickets_by_fi[fi] = r["det_wickets"]
        pads_by_fi[fi] = r["det_pads"]
        batsman_by_fi[fi] = r["det_batsman_box"]
        bats_by_fi[fi] = r["det_bats"]

    # Initialize default LBW overlay
    lbw_overlay = {
        "pitch_inline": False,
        "impact_inline": False,
        "pad_contact": False,
        "wickets_hitting": False,
        "pitch_point": None,
        "impact_point": None,
        "bounce_frame": None,
        "impact_frame_idx": None,
        "stump_intersection": None,
        "fitted_polyline": [],
        "predicted_extension": [],
        "wicket_line": None,
        "decision": "NOT OUT",
        "geometric_lbw": False,
        "reason": "No ball or wicket detections available",
    }
    if n_paths > 0:
        wickets_frames = [wickets_by_fi.get(i, []) for i in range(n_paths)]
        pads_frames = [pads_by_fi.get(i, []) for i in range(n_paths)]
        batsman_frames = [batsman_by_fi.get(i, None) for i in range(n_paths)]
        bats_frames = [bats_by_fi.get(i, []) for i in range(n_paths)]
        lbw_overlay = analyze_lbw_sequence(
            all_ball_infos,
            trajectory_model,
            wickets_frames,
            pads_frames,
            batsman_frames,
            bats_frames,
        )

    frames_dir = None
    if frame_paths:
        frames_dir = os.path.dirname(os.path.abspath(str(frame_paths[0])))
    if frames_dir:
        summary_path = os.path.join(frames_dir, "lbw_summary.json")
        try:
            payload = dict(lbw_overlay)
            payload["api"] = lbw_overlay_for_api(lbw_overlay)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(_sanitize_for_json(payload), f, indent=2)
        except OSError as e:
            print(f"[WARN] Could not write lbw_summary.json: {e}")

    for rec in frame_records:
        frame_path = rec["path"]
        fi = int(rec["metadata"]["frame_index"])
        if fi % progress_step == 0:
            logger.info("Pipeline pass 2/2 progress: %s/%s frames", fi, n_paths)
        clean_frame = cv2.imread(frame_path)
        if clean_frame is None:
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

        metadata = rec["metadata"]
        metadata["lbw"] = lbw_overlay_for_api(lbw_overlay)

        vis_frame = visualize_frame(
            frame.copy(),
            all_ball_infos,
            rec["det_persons"],
            rec["det_batsman_box"],
            rec["det_wickets"],
            rec["det_bats"],
            rec["det_pads"],
            rec["det_pose"],
            fi,
            lbw_overlay=lbw_overlay,
        )

        cv2.imwrite(frame_path, vis_frame)
        meta_path = os.path.splitext(frame_path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(metadata), f, indent=2)
    if video_stem and frames_dir and frame_records:
        safe = _safe_video_stem(str(video_stem))
        parent_dir = os.path.dirname(os.path.abspath(frames_dir))
        lbw_viz_dir = os.path.join(parent_dir, f"lbw-visualiser-{safe}")
        try:
            os.makedirs(lbw_viz_dir, exist_ok=True)
        except OSError as e:
            print(f"[WARN] Could not create LBW visualizer dir {lbw_viz_dir}: {e}")
            lbw_viz_dir = None

        if lbw_viz_dir:
            run_tag = _safe_video_stem(os.path.basename(os.path.abspath(frames_dir)))
            impact_fi = lbw_overlay.get("impact_frame_idx")
            target_rec = None
            if impact_fi is not None:
                for r in frame_records:
                    if int(r["metadata"]["frame_index"]) == int(impact_fi):
                        target_rec = r
                        break

            if target_rec is not None:
                fp = target_rec["path"]
                clean = cv2.imread(fp)
                if clean is not None:
                    if preprocess:
                        rev_frame = preprocess_frame(
                            clean,
                            color_mode=None,
                            enable_deblur=GLOBAL_CONFIG["enable_deblur"],
                            enable_sharpen=GLOBAL_CONFIG["enable_sharpen"],
                            enable_clahe=GLOBAL_CONFIG["enable_clahe"],
                            blur_threshold=GLOBAL_CONFIG["blur_threshold"],
                        )
                    else:
                        rev_frame = clean.copy()
                    fi_rev = int(target_rec["metadata"]["frame_index"])
                    lbw_img = visualize_frame(
                        rev_frame.copy(),
                        all_ball_infos,
                        target_rec["det_persons"],
                        target_rec["det_batsman_box"],
                        target_rec["det_wickets"],
                        target_rec["det_bats"],
                        target_rec["det_pads"],
                        target_rec["det_pose"],
                        fi_rev,
                        lbw_overlay=lbw_overlay,
                    )
                    out_name = f"{safe}_impact_f{fi_rev:06d}_{run_tag}.jpg"
                    out_path = os.path.join(lbw_viz_dir, out_name)
                    api_snap = lbw_overlay_for_api(lbw_overlay)
                    card_bgr = render_lbw_review_card(
                        lbw_img,
                        api_snap,
                        lbw_overlay,
                        frame_index=fi_rev,
                    )
                    card_name = f"lbw_review_card_{run_tag}.jpg"
                    card_path_run = os.path.join(os.path.abspath(frames_dir), card_name)
                    try:
                        cv2.imwrite(out_path, lbw_img)
                        print(f"[INFO] LBW review image: {out_path}")
                        cv2.imwrite(card_path_run, card_bgr)
                        print(f"[INFO] LBW review card: {card_path_run}")
                        card_path_vis = os.path.join(lbw_viz_dir, card_name)
                        cv2.imwrite(card_path_vis, card_bgr)
                        ctx = {
                            "video_stem": safe,
                            "impact_frame_idx": int(impact_fi),
                            "source_run_dir": run_tag,
                            "decision": lbw_overlay.get("decision"),
                            "reason": lbw_overlay.get("reason"),
                            "review_image": out_name,
                            "review_card": card_name,
                        }
                        with open(
                            os.path.join(lbw_viz_dir, f"{safe}_lbw_context_{run_tag}.json"),
                            "w",
                            encoding="utf-8",
                        ) as cf:
                            json.dump(_sanitize_for_json(ctx), cf, indent=2)
                    except OSError as e:
                        print(f"[WARN] Could not write LBW review image: {e}")
                    if display:
                        cv2.imshow("LBW review card", card_bgr)
                        cv2.waitKey(800)
                else:
                    print(f"[WARN] Could not read frame for LBW visualizer: {fp}")
            else:
                print(
                    "[INFO] LBW visualizer: no pad/body impact frame detected; "
                    "skipping single-frame LBW image."
                )

    logger.info("Pipeline completed for %s frames", n_paths)
    cv2.destroyAllWindows()


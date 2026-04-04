import cv2
import json
import os
import re
import numpy as np

from BallDetection.pipeline.ball_detector import detect_ball
from BallDetection.pipeline.trajectory import fit_trajectory
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
from lbw_analyzer import (
    analyze_lbw_sequence,
    build_anchors_from_ball_infos,
    lbw_overlay_for_api,
    missing_clip_overlay,
)
from lbw_review_card import render_lbw_review_card


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
    video_stem=None,
):
    batsman_finder = BatsmanFinder(
        iou_thresh=iou_thresh,
        consec_required=consec_required
    )
    batsman_tracker = BatsmanTracker()
    tracking_active = False

    all_ball_infos = []
    frame_records = []
    n_paths = len(frame_paths)

    for frame_idx, frame_path in enumerate(frame_paths):
        clean_frame = cv2.imread(frame_path)
        if clean_frame is None:
            print(f"[WARN] Could not read {frame_path}")
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

        if wicket_override:
            det_wickets = wicket_override
            metadata["detections"].extend(det_wickets)
        else:
            _, det_wickets = detect_wicket(frame.copy(), conf=wicket_conf)
            if det_wickets:
                metadata["detections"].extend(det_wickets)

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
                    print(f"[INFO] ✅ Batsman confirmed at frame {frame_idx}")
        else:
            ok, bbox = batsman_tracker.update(frame.copy())
            if ok:
                det_batsman_box = list(map(int, bbox))
                metadata["tracking_active"] = True
                metadata["detections"].append({"label": "Batsman", "box": det_batsman_box, "tracked": True})
            else:
                print(f"[WARN] Tracker lost at frame {frame_idx}")
                tracking_active = False
                batsman_finder = BatsmanFinder(iou_thresh=iou_thresh, consec_required=consec_required)
                batsman_tracker = BatsmanTracker()

        if det_batsman_box:
            _, kps = estimate_pose(frame.copy(), bbox=det_batsman_box)
            det_pose.extend(kps)
        else:
            for p in det_persons:
                box = list(p[:4])
                _, kps = estimate_pose(frame.copy(), bbox=box)
                det_pose.extend(kps)

        _, det_pads = detect_pads(frame.copy(), det_pose, conf=pad_conf)
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

    anchors = build_anchors_from_ball_infos(all_ball_infos)
    trajectory_model = fit_trajectory(anchors)

    wickets_by_fi = {}
    pads_by_fi = {}
    batsman_by_fi = {}
    for r in frame_records:
        fi = int(r["metadata"]["frame_index"])
        wickets_by_fi[fi] = r["det_wickets"]
        pads_by_fi[fi] = r["det_pads"]
        batsman_by_fi[fi] = r["det_batsman_box"]

    lbw_overlay = missing_clip_overlay(0)
    if n_paths > 0:
        wickets_frames = [wickets_by_fi.get(i, []) for i in range(n_paths)]
        pads_frames = [pads_by_fi.get(i, []) for i in range(n_paths)]
        batsman_frames = [batsman_by_fi.get(i, None) for i in range(n_paths)]
        lbw_overlay = analyze_lbw_sequence(
            all_ball_infos,
            trajectory_model,
            wickets_frames,
            pads_frames,
            batsman_frames,
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
            lbw_overlay=None,
        )

        cv2.imwrite(frame_path, vis_frame)
        meta_path = os.path.splitext(frame_path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(metadata), f, indent=2)
        if display:
            cv2.imshow("FairPlayReviewSystem", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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

    cv2.destroyAllWindows()

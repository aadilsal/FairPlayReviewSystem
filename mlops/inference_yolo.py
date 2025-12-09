"""YOLO-based inference pipeline for cricket video analysis."""
import time
import json
from pathlib import Path
from typing import Dict, Any
import logging
import mlflow
import numpy as np
import cv2

from .config import load_config
from .video_processor import VideoProcessor
from .model_manager_yolo import YOLOModelManager
from .utils import save_json, ensure_dir

logger = logging.getLogger(__name__)


def run_inference(video_path: Path) -> Dict[str, Any]:
    """
    Run complete YOLO inference pipeline on a video.
    
    Args:
        video_path: Path to uploaded video file
    
    Returns:
        Dictionary with status, predictions, and metadata
    """
    logger.info("INFERENCE PIPELINE STARTED")
    logger.info(f"Video: {video_path.name}")
    
    cfg = load_config()
    ensure_dir(cfg.results_dir)

    
    logger.info("Step 1: Validating video...")
    vp = VideoProcessor()
    ok, reason = vp.validate_video(video_path)
    if not ok:
        logger.error(f"Video validation failed: {reason}")
        return {"status": "error", "message": f"video_invalid:{reason}"}
    logger.info("Video validated")

    # Load YOLO models (ball + batsman)
    logger.info("Step 2: Loading YOLO models from MLflow...")
    mm = YOLOModelManager(cfg.mlflow_tracking_uri, cfg.mlflow_username, cfg.mlflow_password)

    start = time.time()

    # Attempt to load ball model (backwards compatible with MODEL_RUN_ID)
    ball_model = None
    batsman_model = None
    try:
        if cfg.model_run_id_ball:
            ball_model = mm.get_model(cfg.model_run_id_ball)
            logger.info(f"  ✓ Ball model loaded from run {cfg.model_run_id_ball}")
        elif cfg.model_run_id:
            ball_model = mm.get_model(cfg.model_run_id)
            logger.info(f"  ✓ Ball model loaded from run {cfg.model_run_id} (MODEL_RUN_ID)")
    except Exception as e:
        logger.warning(f"  ⚠️ Ball model loading failed: {e}")

    # Attempt to load batsman model if configured
    try:
        if cfg.model_run_id_batsman:
            batsman_model = mm.get_model(cfg.model_run_id_batsman)
            logger.info(f"  ✓ Batsman model loaded from run {cfg.model_run_id_batsman}")
    except Exception as e:
        logger.warning(f"  ⚠️ Batsman model loading failed: {e}")

    if not ball_model and not batsman_model:
        logger.error("  ❌ No models available for inference (ball or batsman)")
        return {"status": "error", "message": "no_models_available"}

    # Extract and preprocess frames
    logger.info("  🎞️  Step 3: Extracting frames...")
    frames = []
    frame_count = 0
    for f in vp.extract_frames(video_path):
        frames.append(f)  # Keep original format for YOLO
        frame_count += 1

    if len(frames) == 0:
        logger.error("  ❌ No frames extracted from video")
        return {"status": "error", "message": "no_frames_extracted"}
    
    logger.info(f"  ✓ Extracted {len(frames)} frames")

    # Run YOLO inference on frames with both models
    logger.info("  🔮 Step 4: Running YOLO object detection (ball + batsman)...")
    all_detections = []

    try:
        for idx, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_detections = []

          
            def process_results(results, src_name, src_model, filter_classes=None):
                dets = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            cls_name = src_model.names[cls] if hasattr(src_model, 'names') else str(cls)
                            if filter_classes and cls_name not in filter_classes:
                                continue  # Skip detections not in filter
                            dets.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": round(conf, 3),
                                "class_id": cls,
                                "class_name": cls_name,
                                "source": src_name,
                            })
                return dets

          
            if ball_model:
                try:
                    results_ball = ball_model(frame_bgr, conf=0.25, iou=0.45, verbose=False, device=cfg.device)
                    bd = process_results(results_ball, 'ball', ball_model, filter_classes=['sports ball'])
                    frame_detections.extend(bd)
                except Exception as e:
                    logger.warning(f"  ⚠️ Ball model inference failed on frame {idx}: {e}")

            # Run batsman model
            if batsman_model:
                try:
                    results_bat = batsman_model(frame_bgr, conf=0.25, iou=0.45, verbose=False, device=cfg.device)
                    bd = process_results(results_bat, 'batsman', batsman_model, filter_classes=['person'])
                    frame_detections.extend(bd)
                except Exception as e:
                    logger.warning(f"  ⚠️ Batsman model inference failed on frame {idx}: {e}")

            detection_result = {
                "frame": idx,
                "num_detections": len(frame_detections),
                "detections": frame_detections[:50]  
            }
            all_detections.append(detection_result)

            if (idx + 1) % 10 == 0:
                logger.info(f"  ⏳ Processed {idx + 1}/{len(frames)} frames...")

        logger.info(f"  ✓ Completed detection on all {len(frames)} frames")
    except Exception as e:
        logger.error(f"  ❌ Inference failed: {e}")
        import traceback
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"inference_failed:{e}"}

    inference_time = time.time() - start
    logger.info(f"  ⏱️  Total inference time: {inference_time:.2f}s")

    # Count total detections and breakdown by source
    total_objects = sum(d["num_detections"] for d in all_detections)
    ball_count = 0
    batsman_count = 0
    for fr in all_detections:
        for det in fr.get("detections", []):
            if det.get("source") == 'ball':
                ball_count += 1
            elif det.get("source") == 'batsman':
                batsman_count += 1
    logger.info(f"  📊 Total objects detected: {total_objects}")

    # Log run to mlflow
    logger.info("  📊 Step 5: Logging to MLflow...")
    run_id = None
    out_path = None
    ball_out_path = None
    try:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri or mlflow.get_tracking_uri())
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"  📝 MLflow Run ID: {run_id}")
            
            mlflow.log_param("model_version", cfg.model_version)
            mlflow.log_param("video_path", str(video_path))
            mlflow.log_param("frames_processed", len(frames))
            mlflow.log_param("framework", "ultralytics_yolo")
            # Log which model run ids were used
            if cfg.model_run_id_ball:
                mlflow.log_param("model_run_id_ball", cfg.model_run_id_ball)
            if cfg.model_run_id_batsman:
                mlflow.log_param("model_run_id_batsman", cfg.model_run_id_batsman)
            mlflow.log_metric("inference_time", inference_time)
            mlflow.log_metric("avg_time_per_frame", inference_time / len(frames))
            mlflow.log_metric("total_detections", total_objects)
            mlflow.log_metric("ball_detections_total", ball_count)
            mlflow.log_metric("batsman_detections_total", batsman_count)
            logger.info("  ✓ Logged parameters and metrics")

            results = {
                "detections": all_detections,
                "total_frames": len(frames),
                "total_objects": total_objects,
                "inference_time": inference_time,
            }

            out_path = cfg.results_dir / f"results_{run_id}.json"
            save_json(results, out_path)
            mlflow.log_artifact(str(out_path), artifact_path="results")
            logger.info(f"  ✓ Saved results to: {out_path.name}")
            logger.info("  ✓ Uploaded artifacts to MLflow")

            # Additionally extract and log ball-specific detections (if model was trained for ball)
            try:
                ball_detections = []
                for fdet in all_detections:
                    frame_idx = fdet.get("frame")
                    for det in fdet.get("detections", []):
                        cls_name = str(det.get("class_name", "")).lower()
                        # match labels that include 'ball' or 'cricket'
                        if "ball" in cls_name or "cricket" in cls_name:
                            det_copy = det.copy()
                            det_copy["frame"] = frame_idx
                            ball_detections.append(det_copy)

                mlflow.log_metric("ball_detections_total", len(ball_detections))
                logger.info(f"  📌 Ball detections found: {len(ball_detections)}")

                if ball_detections:
                    ball_out = {
                        "ball_detections": ball_detections,
                        "total_ball_detections": len(ball_detections),
                        "model_run_id": run_id,
                    }
                    ball_out_path = cfg.results_dir / f"ball_results_{run_id}.json"
                    save_json(ball_out, ball_out_path)
                    mlflow.log_artifact(str(ball_out_path), artifact_path="results/ball_detections")
                    logger.info(f"  ✓ Saved ball-specific results to: {ball_out_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to extract/log ball detections: {e}")
    except Exception as e:
        logger.warning(f"  ⚠️  MLflow logging failed: {e}")
        # non-fatal: continue returning results even if logging fails

    # Prepare preview for frontend: include frames with ball and batsman detections
    ball_frames = [d for d in all_detections if any(det.get("source") == "ball" for det in d.get("detections", []))]
    batsman_frames = [d for d in all_detections if any(det.get("source") == "batsman" for det in d.get("detections", []))]

    return {
        "status": "success",
        "run_id": run_id,
        # For frontend: include a small preview of frames with detections
        "predictions": all_detections[:3],  # Backwards-compatible preview (first 3 frames)
        "predictions_preview": (ball_frames[:5] + batsman_frames[:5])[:10],
        "frames_with_detections": [d.get("frame") for d in all_detections if d.get("num_detections", 0) > 0],
        "total_detections": total_objects,
        "frames_processed": len(frames),
        "inference_time": round(inference_time, 2),
        "result_file": out_path.name if out_path else None,
        "ball_result_file": ball_out_path.name if ball_out_path else None,
    }

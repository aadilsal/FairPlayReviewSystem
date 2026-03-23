from fastapi import UploadFile, HTTPException
from API.utils.file_handler import save_upload_file, delete_file
from API.schemas.detection_schemas import DetectionResult
from API.core.supabase_client import supabase_client, MATCHES_TABLE, DETECTION_RESULTS_TABLE
import sys
import os
import uuid
import time
import json
from pathlib import Path


def _bootstrap_pipeline_import_paths() -> str:
    """Ensure root/project subpaths are available for legacy detector imports."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    required = [
        root,
        os.path.join(root, "BallDetection"),
        os.path.join(root, "BatsmanDetection"),
        os.path.join(root, "WicketDetection"),
        os.path.join(root, "utils"),
    ]
    for p in required:
        if p not in sys.path:
            sys.path.append(p)
    return root

# Safely import detection pipeline
try:
    _PROJECT_ROOT = _bootstrap_pipeline_import_paths()
    from detection_pipeline import process_frames_pipeline
    from utils.frame_extractor import extract_video_frames
    from utils.video_utils import frames_to_video_with_custom_path
    DETECTION_PIPELINE_AVAILABLE = True
except ImportError:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DETECTION_PIPELINE_AVAILABLE = False


def _aggregate_frame_metadata(frames_dir: Path) -> tuple[dict, str]:
    """Read frame-level JSON metadata and compute summary statistics."""
    json_files = sorted(frames_dir.glob("frame_*.json"))
    frame_items = []
    summary = {
        "total_frames": len(json_files),
        "frames_with_ball": 0,
        "frames_with_batsman": 0,
        "frames_with_wicket": 0,
        "frames_with_pose": 0,
        "tracking_active_frames": 0,
    }

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            item = json.load(f)
        frame_items.append(item)

        detections = item.get("detections", [])
        labels = [d.get("label") for d in detections]
        if "Ball" in labels:
            summary["frames_with_ball"] += 1
        if "Batsman" in labels:
            summary["frames_with_batsman"] += 1
        if any(lbl and "Wicket" in lbl for lbl in labels):
            summary["frames_with_wicket"] += 1
        if any(lbl and "Pose" in lbl for lbl in labels):
            summary["frames_with_pose"] += 1
        if item.get("tracking_active"):
            summary["tracking_active_frames"] += 1

    merged = {
        "summary": summary,
        "frames": frame_items,
    }

    merged_path = frames_dir / "detection_metadata_merged.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    return summary, str(merged_path)

class DetectionService:
    @staticmethod
    async def analyze_video(
        match_id: int,
        user_id: int,
        video_file: UploadFile,
        person_conf: float = 0.5,
        bat_conf: float = 0.1,
        iou_thresh: float = 0.05,
        consec_frames: int = 3,
        wicket_conf: float = 0.25,
        fps: int = 30,
    ):
        if not DETECTION_PIPELINE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Detection pipeline not available")

        match_check = supabase_client.table(MATCHES_TABLE).select("id").eq("id", match_id).execute()
        if not match_check.data:
            raise HTTPException(status_code=404, detail="Match not found")

        file_path = save_upload_file(video_file)
        record = None
        started_at = time.perf_counter()

        try:
            create_resp = supabase_client.table(DETECTION_RESULTS_TABLE).insert({
                "match_id": match_id,
                "user_id": user_id,
                "input_video_path": file_path,
                "status": "processing",
            }).execute()
            if create_resp.data:
                record = create_resp.data[0]

            request_id = str(uuid.uuid4())[:8]
            video_name = Path(video_file.filename or "uploaded_video").stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            folder_name = f"{video_name}_{timestamp}_{request_id}"

            frames_dir = Path(_PROJECT_ROOT) / "outputs" / "frames" / folder_name
            frames_dir.mkdir(parents=True, exist_ok=True)

            extract_video_frames(file_path, str(frames_dir), fps)

            frame_files = sorted([p for p in os.listdir(frames_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
            frame_paths = [str(frames_dir / p) for p in frame_files]
            if not frame_paths:
                raise RuntimeError("No frames extracted from uploaded video")

            process_frames_pipeline(
                frame_paths,
                person_conf=person_conf,
                bat_conf=bat_conf,
                iou_thresh=iou_thresh,
                consec_required=consec_frames,
                wicket_conf=wicket_conf,
                display=False,
            )

            output_video_path = frames_dir / f"{video_name}_output.mp4"
            frames_to_video_with_custom_path(str(frames_dir), str(output_video_path), fps)

            summary_stats, merged_metadata_path = _aggregate_frame_metadata(frames_dir)
            processing_time_ms = int((time.perf_counter() - started_at) * 1000)

            # Bridge to frontend DRS expectations (api.types.ts -> AnalyzeVideoResponse)
            is_wicket = summary_stats.get("frames_with_wicket", 0) > 0
            is_ball = summary_stats.get("frames_with_ball", 0) > 0
            
            # Simple heuristic for compatibility layer
            impact = "In-line" if is_ball else "Outside"
            pitch = "In-line" if is_ball else "Outside"
            wickets = "Hitting" if is_wicket else "Missing"
            decision = "OUT" if (is_wicket and is_ball) else "NOT OUT"
            confidence = 0.85 if is_wicket else 0.45

            result_payload = {
                "match_id": match_id,
                "user_id": user_id,
                "status": "completed",
                "processing_time_ms": processing_time_ms,
                "frame_count": len(frame_paths),
                "output_video_path": str(output_video_path),
                "frames_dir": str(frames_dir),
                "metadata_path": merged_metadata_path,
                "summary_stats": summary_stats,
                # Frontend specific fields (AnalyzeVideoResponse)
                "impact": impact,
                "pitch": pitch,
                "wickets": wickets,
                "decision": decision,
                "confidence": confidence
            }

            if record:
                supabase_client.table(DETECTION_RESULTS_TABLE).update({
                    "status": "completed",
                    "output_video_path": str(output_video_path),
                    "metadata_path": merged_metadata_path,
                    "summary_stats": summary_stats,
                    "processing_time_ms": processing_time_ms,
                    "result_data": result_payload # Persist the full DRS payload for deep sync
                }).eq("id", record["id"]).execute()

            return DetectionResult(result=result_payload)
        except Exception as e:
            if record:
                processing_time_ms = int((time.perf_counter() - started_at) * 1000)
                supabase_client.table(DETECTION_RESULTS_TABLE).update({
                    "status": "failed",
                    "error_message": str(e),
                    "processing_time_ms": processing_time_ms,
                }).eq("id", record["id"]).execute()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            delete_file(file_path)

    @staticmethod
    async def detect_ball(video_file: UploadFile):
        file_path = save_upload_file(video_file)
        try:
            result = {"message": "Ball detection logic called"}
            return DetectionResult(result=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            delete_file(file_path)

    @staticmethod
    async def detect_batsman(video_file: UploadFile):
        file_path = save_upload_file(video_file)
        try:
            result = {"message": "Batsman detection logic called"}
            return DetectionResult(result=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            delete_file(file_path)

    @staticmethod
    async def detect_wicket(video_file: UploadFile):
        file_path = save_upload_file(video_file)
        try:
            result = {"message": "Wicket detection logic called"}
            return DetectionResult(result=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            delete_file(file_path)

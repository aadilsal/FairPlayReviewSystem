from fastapi import UploadFile, HTTPException
from API.utils.file_handler import save_upload_file, delete_file
from API.schemas.detection_schemas import DetectionResult
from API.core.supabase_client import (
    supabase_client,
    supabase_admin_client,
    MATCHES_TABLE,
    DETECTION_RESULTS_TABLE,
)
from API.services.prediction_service import PredictionService
from API.schemas.review_schemas import ReviewCreate
from API.services.review_service import ReviewService
from API.services.wicket_config_service import WicketConfigService
import sys
import os
import uuid
import time
import json
import base64
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
    from detection_pipeline import process_frames_pipeline, _safe_video_stem
    from utils.frame_extractor import extract_video_frames
    from utils.video_utils import frames_to_video_with_custom_path
    DETECTION_PIPELINE_AVAILABLE = True
except ImportError:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DETECTION_PIPELINE_AVAILABLE = False


REVIEW_VIDEOS_BUCKET = "review-videos"
REVIEW_STORAGE_SIGNED_URL_SECONDS = 60 * 60  # same TTL as video_proxy / reviews


def _write_client():
    return supabase_admin_client or supabase_client


def _upload_review_video_and_get_object_path(*, local_video_path: Path, user_id: int, match_id: int) -> str:
    if not local_video_path.exists():
        raise RuntimeError(f"Output video not found: {local_video_path}")

    object_path = f"reviews/user_{user_id}/match_{match_id}/{local_video_path.stem}_{uuid.uuid4().hex}.mp4"
    storage = _write_client().storage.from_(REVIEW_VIDEOS_BUCKET)

    payload = local_video_path.read_bytes()
    if not payload:
        raise RuntimeError("Output video is empty")

    try:
        storage.upload(object_path, payload, {"content-type": "video/mp4", "upsert": "true"})
    except TypeError:
        storage.upload(object_path, payload)
    except Exception as exc:
        raise RuntimeError(f"Video upload failed: {exc}") from exc

    # For private buckets we'll store the object path and create signed URLs at read time.
    return object_path


def _upload_lbw_review_card_jpeg(*, local_path: Path, user_id: int, match_id: int) -> str:
    if not local_path.is_file():
        raise RuntimeError(f"LBW review card not found: {local_path}")

    object_path = (
        f"reviews/user_{user_id}/match_{match_id}/lbw_review_card_{uuid.uuid4().hex}.jpg"
    )
    storage = _write_client().storage.from_(REVIEW_VIDEOS_BUCKET)
    payload = local_path.read_bytes()
    if not payload:
        raise RuntimeError("LBW review card file is empty")

    try:
        storage.upload(
            object_path,
            payload,
            {"content-type": "image/jpeg", "upsert": "true"},
        )
    except TypeError:
        storage.upload(object_path, payload)
    except Exception as exc:
        raise RuntimeError(f"LBW review card upload failed: {exc}") from exc

    return object_path


def _sign_review_storage_object(object_path: str) -> str:
    storage = _write_client().storage.from_(REVIEW_VIDEOS_BUCKET)
    signed = storage.create_signed_url(object_path, REVIEW_STORAGE_SIGNED_URL_SECONDS)
    return ReviewService._resolve_signed_url(signed)


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


def _lbw_review_card_path(frames_dir: Path) -> Path:
    """Same naming as detection_pipeline LBW card output."""
    run_tag = _safe_video_stem(frames_dir.resolve().name)
    return frames_dir / f"lbw_review_card_{run_tag}.jpg"


def _read_jpeg_as_data_url(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class DetectionService:
    @staticmethod
    def _build_wicket_override(match_id: int, user_id: int):
        try:
            cfg = WicketConfigService.get_config(match_id, user_id)
        except HTTPException:
            # If match doesn't exist or is not owned, allow normal error path later.
            return None
        if not cfg:
            return None
        if not cfg.get("configured"):
            return None
        near_box = cfg.get("near_box")
        far_box = cfg.get("far_box")
        # Configured is driven by FAR wicket presence; near is optional.
        if not far_box:
            return None
        override = [{"label": "Wicket_Far", "box": far_box, "conf": 1.0, "source": "configured"}]
        if near_box:
            override.insert(0, {"label": "Wicket_Near", "box": near_box, "conf": 1.0, "source": "configured"})
        return override

    @staticmethod
    async def analyze_video(
        match_id: int,
        user_id: int,
        original_decision: str | None,
        video_file: UploadFile,
        person_conf: float = 0.5,
        bat_conf: float = 0.1,
        pad_conf: float = 0.1,
        iou_thresh: float = 0.05,
        consec_frames: int = 3,
        wicket_conf: float = 0.25,
        preprocess: bool = True,
        fps: int = 30,
        display: bool = True,
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

            wicket_override = DetectionService._build_wicket_override(match_id, user_id)

            process_frames_pipeline(
                frame_paths,
                person_conf=person_conf,
                bat_conf=bat_conf,
                pad_conf=pad_conf,
                iou_thresh=iou_thresh,
                consec_required=consec_frames,
                wicket_conf=wicket_conf,
                preprocess=preprocess,
                display=display,
                wicket_override=wicket_override,
                video_stem=video_name,
            )

            output_video_path = frames_dir / f"{video_name}_output.mp4"
            frames_to_video_with_custom_path(str(frames_dir), str(output_video_path), fps)

            summary_stats, merged_metadata_path = _aggregate_frame_metadata(frames_dir)
            processing_time_ms = int((time.perf_counter() - started_at) * 1000)

            # Trajectory prediction -> DRS-compatible fields.
            prediction = PredictionService.predict_from_frames(frames_dir)
            impact = prediction["impact"]
            pitch = prediction["pitch"]
            wickets = prediction["wickets"]
            decision = prediction["decision"]
            confidence = prediction["confidence"]

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
                # Extra debugging/inspection fields (safe for frontend).
                "trajectory_prediction": prediction,
                # Frontend specific fields (AnalyzeVideoResponse)
                "impact": impact,
                "pitch": pitch,
                "wickets": wickets,
                "decision": decision,
                "confidence": confidence,
            }
            if original_decision is not None:
                result_payload["original_decision"] = original_decision

            card_path = _lbw_review_card_path(frames_dir)
            card_data_url = _read_jpeg_as_data_url(card_path)
            result_payload["lbw_review_card_image"] = card_data_url
            result_payload["lbw_review_card_filename"] = (
                card_path.name if card_data_url else None
            )
            result_payload["lbw_review_card_object_path"] = None
            result_payload["lbw_review_card_url"] = None

            # Upload output video to Supabase Storage (store object path; signed URLs are generated when fetching reviews).
            output_video_object_path = None
            output_video_upload_error = None
            try:
                output_video_object_path = _upload_review_video_and_get_object_path(
                    local_video_path=Path(output_video_path),
                    user_id=user_id,
                    match_id=match_id,
                )
                result_payload["output_video_object_path"] = output_video_object_path
            except Exception as e:
                output_video_upload_error = str(e)
                result_payload["output_video_upload_error"] = output_video_upload_error

            if card_path.is_file():
                try:
                    card_object_path = _upload_lbw_review_card_jpeg(
                        local_path=card_path,
                        user_id=user_id,
                        match_id=match_id,
                    )
                    result_payload["lbw_review_card_object_path"] = card_object_path
                    signed_card = _sign_review_storage_object(card_object_path)
                    if signed_card:
                        result_payload["lbw_review_card_url"] = signed_card
                except Exception as e:
                    result_payload["lbw_review_card_upload_error"] = str(e)

            # Persist a user-visible "review" record for later viewing.
            saved_review = None
            review_save_error = None
            try:
                review_create = ReviewCreate(
                    match_id=match_id,
                    original_decision=original_decision,
                    decision=decision,
                    impact=impact,
                    pitch=pitch,
                    wickets=wickets,
                    video_uri=output_video_object_path or str(output_video_path),
                    lbw_review_card_uri=result_payload.get("lbw_review_card_object_path"),
                    content=merged_metadata_path,
                    analysis=json.dumps(prediction),
                )
                saved_review = await ReviewService.create_review(review_create, user_id)
            except Exception as e:
                # Don't discard the analysis output if review persistence fails.
                review_save_error = str(e)

            result_payload["review_saved"] = bool(saved_review)
            result_payload["review"] = saved_review
            if review_save_error:
                result_payload["review_save_error"] = review_save_error

            if record:
                result_for_db = dict(result_payload)
                # Avoid storing multi‑MB base64 blobs in JSON columns.
                if "lbw_review_card_image" in result_for_db:
                    result_for_db["lbw_review_card_image"] = None
                    result_for_db["lbw_review_card_image_omitted"] = bool(
                        result_payload.get("lbw_review_card_image")
                    )
                # Signed URLs expire; keep only the storage object path in JSON.
                result_for_db.pop("lbw_review_card_url", None)
                supabase_client.table(DETECTION_RESULTS_TABLE).update({
                    "status": "completed",
                    "output_video_path": str(output_video_path),
                    "metadata_path": merged_metadata_path,
                    "summary_stats": summary_stats,
                    "processing_time_ms": processing_time_ms,
                    "result_data": result_for_db,
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

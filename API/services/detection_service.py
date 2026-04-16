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
from API.utils.lbw_decision import resolve_final_lbw_decision, sanitize_prediction_decisions
from API.schemas.review_schemas import ReviewCreate
from API.services.review_service import ReviewService
from API.services.wicket_config_service import WicketConfigService
from API.services.snick_detection_service import SnickDetectionService, AudioAnalysisConfig
from API.core.config import settings
from utils.audio_extractor import extract_audio_to_wav, AudioExtractionError
from global_config import GLOBAL_CONFIG
import sys
import os
import uuid
import time
import json
import base64
from pathlib import Path
import logging


logger = logging.getLogger("fairplay.api.detection")


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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _ball_center_from_detection(detections: list[dict]) -> tuple[float, float] | None:
    for d in detections:
        if d.get("label") != "Ball":
            continue
        data = d.get("data") or {}
        ip = data.get("interpolated_position")
        if isinstance(ip, (list, tuple)) and len(ip) >= 2:
            return float(ip[0]), float(ip[1])
        box = data.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x, y, w, h = box
            return float(x + w / 2.0), float(y + h / 2.0)
    return None


def _distance_point_to_box(px: float, py: float, box: list[float]) -> float:
    x, y, w, h = [float(v) for v in box]
    nx = min(max(px, x), x + w)
    ny = min(max(py, y), y + h)
    return float(((px - nx) ** 2 + (py - ny) ** 2) ** 0.5)


def _estimate_visual_contact_score(frames_dir: Path, impact_frame_idx: int | None) -> tuple[float, dict]:
    if impact_frame_idx is None:
        return 0.0, {"reason": "impact_frame_missing"}

    best_score = 0.0
    best_frame = None
    best_distance = None

    for fi in range(max(0, impact_frame_idx - 3), impact_frame_idx + 4):
        frame_json = frames_dir / f"frame_{fi:06d}.json"
        if not frame_json.exists():
            continue
        try:
            with open(frame_json, "r", encoding="utf-8") as f:
                frame_meta = json.load(f)
        except Exception:
            continue

        detections = frame_meta.get("detections", [])
        center = _ball_center_from_detection(detections)
        if center is None:
            continue

        bat_boxes = [d.get("box") for d in detections if d.get("label") == "Bat" and isinstance(d.get("box"), list)]
        if not bat_boxes:
            continue

        px, py = center
        min_dist = min(_distance_point_to_box(px, py, b) for b in bat_boxes)
        spatial = _clamp01(1.0 - (min_dist / 40.0))
        temporal = _clamp01(1.0 - (abs(fi - impact_frame_idx) / 5.0))
        score = spatial * (0.7 + 0.3 * temporal)
        if score > best_score:
            best_score = score
            best_frame = fi
            best_distance = min_dist

    return float(best_score), {
        "best_frame": best_frame,
        "best_distance_px": best_distance,
    }


def _fuse_snick_scores(
    *,
    visual_score: float,
    audio_result: dict,
    impact_frame_idx: int | None,
    fps: int,
) -> dict:
    if audio_result.get("status") != "ok":
        return {
            "status": "unavailable",
            "snick_detected": False,
            "snick_confidence": None,
            "snick_timestamp_ms": None,
            "reason": audio_result.get("reason") or "audio_unavailable",
            "visual_score": float(visual_score),
            "audio_score": None,
            "fused_score": None,
        }

    audio_score = float(audio_result.get("audio_confidence") or 0.0)
    event_ts = audio_result.get("best_event_timestamp_ms")
    target_ts = None
    if impact_frame_idx is not None and fps > 0:
        target_ts = (float(impact_frame_idx) / float(fps)) * 1000.0

    align_window_ms = int(getattr(settings, "SNICK_ALIGN_WINDOW_MS", GLOBAL_CONFIG.get("snick_align_window_ms", 80)))
    if target_ts is not None and event_ts is not None:
        if abs(float(event_ts) - target_ts) > float(align_window_ms):
            audio_score *= 0.35

    wv = float(getattr(settings, "SNICK_VISUAL_WEIGHT", GLOBAL_CONFIG.get("snick_visual_weight", 0.45)))
    wa = float(getattr(settings, "SNICK_AUDIO_WEIGHT", GLOBAL_CONFIG.get("snick_audio_weight", 0.55)))
    den = max(1e-6, wv + wa)
    fused = _clamp01((wv * float(visual_score) + wa * float(audio_score)) / den)

    detect_threshold = float(
        getattr(settings, "SNICK_DETECT_THRESHOLD", GLOBAL_CONFIG.get("snick_detect_threshold", 0.62))
    )
    detected = bool(fused >= detect_threshold and float(visual_score) >= 0.25)

    return {
        "status": "ok",
        "snick_detected": detected,
        "snick_confidence": float(fused),
        "snick_timestamp_ms": float(event_ts) if event_ts is not None else None,
        "reason": None,
        "visual_score": float(visual_score),
        "audio_score": float(audio_score),
        "fused_score": float(fused),
        "audio_event_count": int(audio_result.get("candidate_event_count") or 0),
        "audio_top_events": audio_result.get("top_events") or [],
    }


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
        extracted_audio_path = None
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

            # Extract audio once and reuse for snick analysis after visual impact frame is known.
            try:
                extracted_audio_path = extract_audio_to_wav(
                    file_path,
                    output_wav_path=str(frames_dir / f"{video_name}_audio.wav"),
                    sample_rate=int(getattr(settings, "SNICK_AUDIO_SAMPLE_RATE", 16000)),
                    ffmpeg_binary=(getattr(settings, "FFMPEG_BINARY", "") or None),
                )
                logger.info("[analyze_video] Extracted audio for snick analysis: %s", extracted_audio_path)
            except AudioExtractionError as exc:
                logger.warning("[analyze_video] Audio extraction unavailable: %s", exc)
                extracted_audio_path = None

            extract_video_frames(file_path, str(frames_dir), fps)

            frame_files = sorted([p for p in os.listdir(frames_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
            frame_paths = [str(frames_dir / p) for p in frame_files]
            if not frame_paths:
                raise RuntimeError("No frames extracted from uploaded video")

            logger.info(
                "[analyze_video] Extracted %s frames for match_id=%s user_id=%s",
                len(frame_paths),
                match_id,
                user_id,
            )

            wicket_override = DetectionService._build_wicket_override(match_id, user_id)

            pipeline_started = time.perf_counter()
            logger.info(
                "[analyze_video] Starting detection pipeline match_id=%s user_id=%s display=%s",
                match_id,
                user_id,
                display,
            )

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

            logger.info(
                "[analyze_video] Detection pipeline finished in %.2fs for match_id=%s",
                time.perf_counter() - pipeline_started,
                match_id,
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
            raw_decision = prediction["decision"]
            decision, review_outcome, normalized_original_decision = resolve_final_lbw_decision(
                raw_decision,
                original_decision,
            )
            prediction = sanitize_prediction_decisions(prediction, decision)
            confidence = prediction["confidence"]

            impact_frame_idx = prediction.get("impact_frame_idx")
            visual_score, visual_diag = _estimate_visual_contact_score(frames_dir, impact_frame_idx)

            if extracted_audio_path:
                try:
                    audio_result = SnickDetectionService.analyze(
                        extracted_audio_path,
                        fps=fps,
                        preferred_frame_idx=impact_frame_idx,
                        config=AudioAnalysisConfig(
                            low_hz=int(getattr(settings, "SNICK_LOW_HZ", GLOBAL_CONFIG.get("snick_low_hz", 1200))),
                            high_hz=int(getattr(settings, "SNICK_HIGH_HZ", GLOBAL_CONFIG.get("snick_high_hz", 6500))),
                            peak_prominence=float(
                                getattr(settings, "SNICK_PEAK_PROMINENCE", GLOBAL_CONFIG.get("snick_peak_prominence", 2.5))
                            ),
                            align_window_ms=int(
                                getattr(settings, "SNICK_ALIGN_WINDOW_MS", GLOBAL_CONFIG.get("snick_align_window_ms", 80))
                            ),
                        ),
                    )
                except Exception as exc:
                    logger.warning("[analyze_video] Snick analysis failed, using fallback: %s", exc)
                    audio_result = {
                        "status": "unavailable",
                        "reason": f"analysis_failed: {exc}",
                        "snick_detected": False,
                        "audio_confidence": 0.0,
                    }
            else:
                audio_result = {
                    "status": "unavailable",
                    "reason": "ffmpeg_unavailable_or_extraction_failed",
                    "snick_detected": False,
                    "audio_confidence": 0.0,
                }

            snick_result = _fuse_snick_scores(
                visual_score=visual_score,
                audio_result=audio_result,
                impact_frame_idx=impact_frame_idx,
                fps=fps,
            )

            low_threshold = float(
                getattr(settings, "SNICK_LOW_THRESHOLD", GLOBAL_CONFIG.get("snick_low_threshold", 0.30))
            )
            if snick_result.get("status") == "ok":
                fused = float(snick_result.get("fused_score") or 0.0)
                if snick_result.get("snick_detected"):
                    confidence = _clamp01(float(confidence) + 0.12 * fused)
                elif fused < low_threshold and visual_score > 0.45:
                    confidence = _clamp01(float(confidence) - 0.05)
                prediction["confidence"] = confidence

            prediction["snick"] = {
                "status": snick_result.get("status"),
                "detected": snick_result.get("snick_detected"),
                "confidence": snick_result.get("snick_confidence"),
                "timestamp_ms": snick_result.get("snick_timestamp_ms"),
                "reason": snick_result.get("reason"),
                "visual_score": snick_result.get("visual_score"),
                "audio_score": snick_result.get("audio_score"),
                "fused_score": snick_result.get("fused_score"),
                "visual_diagnostics": visual_diag,
            }

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
                "snick_detected": snick_result.get("snick_detected"),
                "snick_confidence": snick_result.get("snick_confidence"),
                "snick_timestamp_ms": snick_result.get("snick_timestamp_ms"),
                "snick_status": snick_result.get("status"),
                "snick_unavailable_reason": snick_result.get("reason"),
            }
            if normalized_original_decision is not None:
                result_payload["original_decision"] = normalized_original_decision
            if review_outcome:
                result_payload["review_outcome"] = review_outcome

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
                    original_decision=normalized_original_decision,
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
            if extracted_audio_path:
                delete_file(extracted_audio_path)

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

from fastapi import APIRouter, UploadFile, File, Depends
import logging
from API.services.detection_service import DetectionService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()
logger = logging.getLogger("fairplay.api.detection")

@router.post("/analyze-video")
async def analyze_video(
    match_id: int,
    video_file: UploadFile = File(...),
    person_conf: float = 0.5,
    bat_conf: float = 0.1,
    iou_thresh: float = 0.05,
    consec_frames: int = 3,
    wicket_conf: float = 0.25,
    fps: int = 30,
    current_user=Depends(get_current_user),
):
    logger.info(
        "Analyze video request by user_id=%s for match_id=%s file=%s",
        current_user["id"],
        match_id,
        video_file.filename
    )
    result = await DetectionService.analyze_video(
        match_id=match_id,
        user_id=current_user["id"],
        video_file=video_file,
        person_conf=person_conf,
        bat_conf=bat_conf,
        iou_thresh=iou_thresh,
        consec_frames=consec_frames,
        wicket_conf=wicket_conf,
        fps=fps,
    )
    logger.info("Analyze video completed for match_id=%s", match_id)
    return success_response(data=result.result, message="Video analyzed")

@router.post("/detect/ball")
async def detect_ball(video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    logger.info("Ball detection request by user_id=%s file=%s", current_user["id"], video_file.filename)
    result = await DetectionService.detect_ball(video_file)
    logger.info("Ball detection completed for user_id=%s", current_user["id"])
    return success_response(data=result.result, message="Ball detection complete")

@router.post("/detect/batsman")
async def detect_batsman(video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    logger.info("Batsman detection request by user_id=%s file=%s", current_user["id"], video_file.filename)
    result = await DetectionService.detect_batsman(video_file)
    logger.info("Batsman detection completed for user_id=%s", current_user["id"])
    return success_response(data=result.result, message="Batsman detection complete")

@router.post("/detect/wicket")
async def detect_wicket(video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    logger.info("Wicket detection request by user_id=%s file=%s", current_user["id"], video_file.filename)
    result = await DetectionService.detect_wicket(video_file)
    logger.info("Wicket detection completed for user_id=%s", current_user["id"])
    return success_response(data=result.result, message="Wicket detection complete")

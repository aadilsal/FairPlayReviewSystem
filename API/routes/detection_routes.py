from fastapi import APIRouter, UploadFile, File, Depends
from API.services.detection_service import DetectionService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()

@router.post("/analyze-video")
async def analyze_video(match_id: int, video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    result = await DetectionService.analyze_video(match_id, video_file)
    return success_response(data=result.result, message="Video analyzed")

@router.post("/detect/ball")
async def detect_ball(video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    result = await DetectionService.detect_ball(video_file)
    return success_response(data=result.result, message="Ball detection complete")

@router.post("/detect/batsman")
async def detect_batsman(video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    result = await DetectionService.detect_batsman(video_file)
    return success_response(data=result.result, message="Batsman detection complete")

@router.post("/detect/wicket")
async def detect_wicket(video_file: UploadFile = File(...), current_user=Depends(get_current_user)):
    result = await DetectionService.detect_wicket(video_file)
    return success_response(data=result.result, message="Wicket detection complete")

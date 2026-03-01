from fastapi import UploadFile, HTTPException
from API.utils.file_handler import save_upload_file, delete_file
from API.schemas.detection_schemas import DetectionResult
import sys
import os

# Safely import detection pipeline
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from detection_pipeline import run_detection_pipeline
    DETECTION_PIPELINE_AVAILABLE = True
except ImportError:
    DETECTION_PIPELINE_AVAILABLE = False

class DetectionService:
    @staticmethod
    async def analyze_video(match_id: int, video_file: UploadFile):
        file_path = save_upload_file(video_file)
        try:
            if DETECTION_PIPELINE_AVAILABLE:
                result = run_detection_pipeline(file_path, match_id)
                return DetectionResult(result=result)
            else:
                return DetectionResult(result={"message": "Detection pipeline not available", "match_id": match_id})
        except Exception as e:
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

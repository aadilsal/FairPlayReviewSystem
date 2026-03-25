from pydantic import BaseModel
from typing import Optional

class VideoAnalysisRequest(BaseModel):
    match_id: int
    video_file: Optional[str]  # Path or filename

class DetectionResult(BaseModel):
    result: dict
    message: Optional[str] = None

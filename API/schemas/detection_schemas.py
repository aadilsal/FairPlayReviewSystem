from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class VideoAnalysisRequest(BaseModel):
    match_id: int
    person_conf: float = Field(default=0.5, ge=0.1, le=1.0)
    bat_conf: float = Field(default=0.1, ge=0.05, le=1.0)
    iou_thresh: float = Field(default=0.05, ge=0.0, le=1.0)
    consec_frames: int = Field(default=3, ge=1, le=10)
    wicket_conf: float = Field(default=0.25, ge=0.1, le=1.0)
    fps: int = Field(default=30, ge=1, le=120)

class DetectionResult(BaseModel):
    result: Dict[str, Any]
    message: Optional[str] = None

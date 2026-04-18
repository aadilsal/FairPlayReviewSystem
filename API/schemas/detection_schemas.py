from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal


LbwDecision = Literal["OUT", "NOT OUT"]

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


class AnalyzeVideoResult(BaseModel):
    decision: LbwDecision
    original_decision: Optional[LbwDecision] = None
    confidence: Optional[float] = None
    review_outcome: Optional[Literal["inconclusive"]] = None
    snick_detected: Optional[bool] = None
    snick_confidence: Optional[float] = None
    snick_timestamp_ms: Optional[float] = None
    snick_status: Optional[str] = None
    snick_unavailable_reason: Optional[str] = None

    class Config:
        extra = "allow"


class AnalyzeVideoSuccessResponse(BaseModel):
    status: Literal["success"]
    data: AnalyzeVideoResult
    message: str

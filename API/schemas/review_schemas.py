from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


LbwDecision = Literal["OUT", "NOT OUT"]

class ReviewBase(BaseModel):
    match_id: int
    match_name: Optional[str] = None
    user_id: Optional[int] = None
    delivery: Optional[str] = None
    over: Optional[str] = None
    original_decision: Optional[LbwDecision] = None
    decision: Optional[LbwDecision] = None
    impact: Optional[str] = None
    pitch: Optional[str] = None
    wickets: Optional[str] = None
    video_uri: Optional[str] = None
    lbw_review_card_uri: Optional[str] = None
    content: Optional[str] = None
    analysis: Optional[str] = None

class ReviewCreate(ReviewBase):
    pass

class ReviewUpdate(BaseModel):
    delivery: Optional[str] = None
    over: Optional[str] = None
    original_decision: Optional[LbwDecision] = None
    decision: Optional[LbwDecision] = None
    impact: Optional[str] = None
    pitch: Optional[str] = None
    wickets: Optional[str] = None
    video_uri: Optional[str] = None
    lbw_review_card_uri: Optional[str] = None
    content: Optional[str] = None
    analysis: Optional[str] = None

class ReviewOut(ReviewBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

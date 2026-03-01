from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ReviewBase(BaseModel):
    match_id: int
    user_id: int
    content: str

class ReviewCreate(ReviewBase):
    pass

class ReviewUpdate(BaseModel):
    content: Optional[str]
    analysis: Optional[str]

class ReviewOut(ReviewBase):
    id: int
    analysis: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True

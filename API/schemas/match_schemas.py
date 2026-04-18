from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class MatchBase(BaseModel):
    name: str
    teams: str
    venue: Optional[str] = None
    date: str
    status: str = "upcoming"

class MatchCreate(MatchBase):
    pass

class MatchUpdate(BaseModel):
    name: Optional[str] = None
    teams: Optional[str] = None
    venue: Optional[str] = None
    date: Optional[str] = None
    status: Optional[str] = None

class MatchOut(MatchBase):
    id: int
    user_id: int
    completed_by_system: Optional[bool] = False
    auto_completed_at: Optional[datetime] = None
    completion_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

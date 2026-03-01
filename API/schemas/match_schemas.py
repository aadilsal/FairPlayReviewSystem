from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class MatchBase(BaseModel):
    team_a: str
    team_b: str
    date: datetime
    status: str

class MatchCreate(MatchBase):
    pass

class MatchUpdate(BaseModel):
    team_a: Optional[str]
    team_b: Optional[str]
    date: Optional[datetime]
    status: Optional[str]

class MatchOut(MatchBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

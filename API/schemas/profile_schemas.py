from pydantic import BaseModel, EmailStr
from typing import Optional

class ProfileUpdate(BaseModel):
    username: Optional[str]
    email: Optional[EmailStr]
    avatar: Optional[str]

class ProfileOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    avatar: Optional[str]

    class Config:
        orm_mode = True

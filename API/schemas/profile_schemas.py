from pydantic import BaseModel, EmailStr
from typing import Optional

class ProfileUpdate(BaseModel):
    fname: Optional[str] = None
    lname: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    avatar: Optional[str] = None

class ProfileOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    avatar: Optional[str] = None

    class Config:
        from_attributes = True

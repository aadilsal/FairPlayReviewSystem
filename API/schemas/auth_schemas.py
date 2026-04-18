from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    name: Optional[str] = None
    email: EmailStr

class UserCreate(UserBase):
    name: Optional[str] = None      # optional in payload; service derives from fname/lname when present
    fname: Optional[str] = None     # optional legacy split fields
    lname: Optional[str] = None
    avatar: Optional[str] = None
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(UserBase):
    id: int
    avatar: Optional[str] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class AuthResponseData(BaseModel):
    user: UserOut
    access_token: str
    token_type: str = "bearer"

class ChangePassword(BaseModel):
    email: Optional[EmailStr] = None
    old_password: str
    new_password: str

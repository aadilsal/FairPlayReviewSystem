from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class NotificationBase(BaseModel):
    user_id: int
    message: str


class NotificationCreate(NotificationBase):
    pass


class NotificationOut(NotificationBase):
    id: int
    read: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class NotificationSettingsBase(BaseModel):
    match_alerts: bool = True
    review_updates: bool = True
    system_notifications: bool = True

class NotificationSettingsUpdate(BaseModel):
    match_alerts: Optional[bool] = None
    review_updates: Optional[bool] = None
    system_notifications: Optional[bool] = None

class NotificationSettingsOut(NotificationSettingsBase):
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True

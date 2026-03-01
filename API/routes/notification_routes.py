from fastapi import APIRouter, Depends
from API.services.notification_service import NotificationService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()

@router.get("/")
async def get_notifications(current_user=Depends(get_current_user)):
    notifications = await NotificationService.get_notifications(current_user["id"])
    return success_response(data=notifications)

@router.post("/read")
async def mark_read(notification_id: int):
    notification = await NotificationService.mark_read(notification_id)
    return success_response(data=notification, message="Notification marked as read")

from fastapi import APIRouter, Depends
import logging
from API.services.notification_service import NotificationService
from API.schemas.notification_schemas import NotificationSettingsUpdate
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()
logger = logging.getLogger("fairplay.api.notifications")

@router.get("/")
async def get_notifications(current_user=Depends(get_current_user)):
    logger.info("Get notifications request for user_id=%s", current_user["id"])
    notifications = await NotificationService.get_notifications(current_user["id"])
    logger.info("Get notifications completed for user_id=%s", current_user["id"])
    return success_response(data=notifications)

@router.get("/settings")
async def get_settings(current_user=Depends(get_current_user)):
    logger.info("Get notification settings for user_id=%s", current_user["id"])
    settings = await NotificationService.get_settings(current_user["id"])
    return success_response(data=settings)

@router.put("/settings")
async def update_settings(data: NotificationSettingsUpdate, current_user=Depends(get_current_user)):
    logger.info("Update notification settings for user_id=%s", current_user["id"])
    updated = await NotificationService.update_settings(current_user["id"], data)
    return success_response(data=updated, message="Notification settings updated")

@router.post("/read")
async def mark_read(notification_id: int):
    logger.info("Mark notification as read request for notification_id=%s", notification_id)
    notification = await NotificationService.mark_read(notification_id)
    logger.info("Notification marked as read for notification_id=%s", notification_id)
    return success_response(data=notification, message="Notification marked as read")

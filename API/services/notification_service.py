from fastapi import HTTPException
import logging
from API.core.supabase_client import (
    supabase_client,
    supabase_admin_client,
    NOTIFICATIONS_TABLE,
    NOTIFICATION_SETTINGS_TABLE,
)
from API.schemas.notification_schemas import NotificationCreate, NotificationSettingsUpdate

logger = logging.getLogger("fairplay.api.notifications")


def _get_write_client():
    """Use service-role client for writes to bypass RLS when app JWT is used."""
    return supabase_admin_client or supabase_client

class NotificationService:
    @staticmethod
    async def get_notifications(user_id: int):
        try:
            logger.debug("[get_notifications] Fetching for user_id=%s", user_id)
            response = supabase_client.table(NOTIFICATIONS_TABLE).select("*").eq("user_id", user_id).execute()
            logger.debug("[get_notifications] Returned %d rows for user_id=%s", len(response.data) if response.data else 0, user_id)
            return response.data
        except Exception as e:
            logger.exception("[get_notifications] ERROR user_id=%s | %s", user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_settings(user_id: int):
        try:
            logger.debug("[get_settings] Fetching notification settings for user_id=%s", user_id)
            read_client = _get_write_client()
            response = read_client.table(NOTIFICATION_SETTINGS_TABLE).select("*").eq("user_id", user_id).execute()
            if response.data:
                logger.debug("[get_settings] Found existing settings for user_id=%s", user_id)
                return response.data[0]

            logger.info("[get_settings] No settings row for user_id=%s — creating defaults", user_id)
            default_settings = {
                "user_id": user_id,
                "match_alerts": True,
                "review_updates": True,
                "system_notifications": True
            }
            write_client = _get_write_client()
            insert_resp = write_client.table(NOTIFICATION_SETTINGS_TABLE).insert(default_settings).execute()
            if insert_resp.data:
                logger.info("[get_settings] Default settings created for user_id=%s", user_id)
                return insert_resp.data[0]
            logger.error("[get_settings] Insert default settings returned empty for user_id=%s", user_id)
            raise HTTPException(status_code=500, detail="Failed to initialize notification settings")
        except HTTPException:
            raise
        except Exception as e:
            if "row-level security policy" in str(e).lower() and supabase_admin_client is None:
                logger.error(
                    "[get_settings] RLS blocked write and SUPABASE_SERVICE_ROLE_KEY is not configured. "
                    "Set service role key in backend .env to allow server-side notification writes."
                )
            logger.exception("[get_settings] ERROR user_id=%s | %s", user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_settings(user_id: int, data: NotificationSettingsUpdate):
        try:
            logger.debug("[update_settings] user_id=%s fields=%s", user_id, data.dict(exclude_unset=True))
            await NotificationService.get_settings(user_id)  # ensure row exists

            update_data = data.dict(exclude_unset=True)
            write_client = _get_write_client()
            response = write_client.table(NOTIFICATION_SETTINGS_TABLE).update(update_data).eq("user_id", user_id).execute()
            if response.data:
                logger.info("[update_settings] Updated settings for user_id=%s", user_id)
                return response.data[0]
            logger.error("[update_settings] Update returned empty data for user_id=%s", user_id)
            raise HTTPException(status_code=500, detail="Failed to update notification settings")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[update_settings] ERROR user_id=%s | %s", user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def mark_read(notification_id: int):
        try:
            logger.debug("[mark_read] notification_id=%s", notification_id)
            read_client = _get_write_client()
            response = read_client.table(NOTIFICATIONS_TABLE).select("*").eq("id", notification_id).execute()
            if not response.data:
                logger.warning("[mark_read] Not found notification_id=%s", notification_id)
                raise HTTPException(status_code=404, detail="Notification not found")

            write_client = _get_write_client()
            update_response = write_client.table(NOTIFICATIONS_TABLE).update({"read": True}).eq("id", notification_id).execute()
            if update_response.data:
                logger.info("[mark_read] Marked read notification_id=%s", notification_id)
                return update_response.data[0]
            logger.error("[mark_read] Update returned empty data for notification_id=%s", notification_id)
            raise HTTPException(status_code=500, detail="Failed to mark notification as read")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[mark_read] ERROR notification_id=%s | %s", notification_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def create_notification(data: NotificationCreate):
        try:
            notification_dict = data.dict()
            logger.info("[create_notification] user_id=%s type=%s", notification_dict.get("user_id"), notification_dict.get("type"))
            write_client = _get_write_client()
            response = write_client.table(NOTIFICATIONS_TABLE).insert(notification_dict).execute()
            if response.data:
                logger.info("[create_notification] Created id=%s", response.data[0].get("id"))
                return response.data[0]
            logger.error("[create_notification] Insert returned empty data | payload=%s", notification_dict)
            raise HTTPException(status_code=500, detail="Failed to create notification")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[create_notification] ERROR | %s", e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def delete_notification(notification_id: int):
        try:
            read_client = _get_write_client()
            check = read_client.table(NOTIFICATIONS_TABLE).select("*").eq("id", notification_id).execute()
            if not check.data:
                logger.warning("[delete_notification] Not found id=%s", notification_id)
                raise HTTPException(status_code=404, detail="Notification not found")
            write_client = _get_write_client()
            write_client.table(NOTIFICATIONS_TABLE).delete().eq("id", notification_id).execute()
            logger.info("[delete_notification] Deleted id=%s", notification_id)
            return True
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[delete_notification] ERROR notification_id=%s | %s", notification_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

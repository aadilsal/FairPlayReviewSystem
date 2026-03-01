from fastapi import HTTPException
from API.core.supabase_client import supabase_client, NOTIFICATIONS_TABLE
from API.schemas.notification_schemas import NotificationCreate

class NotificationService:
    @staticmethod
    async def get_notifications(user_id: int):
        try:
            response = supabase_client.table(NOTIFICATIONS_TABLE).select("*").eq("user_id", user_id).execute()
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def mark_read(notification_id: int):
        try:
            response = supabase_client.table(NOTIFICATIONS_TABLE).select("*").eq("id", notification_id).execute()
            if not response.data:
                raise HTTPException(status_code=404, detail="Notification not found")
            
            update_response = supabase_client.table(NOTIFICATIONS_TABLE).update({"read": True}).eq("id", notification_id).execute()
            if update_response.data:
                return update_response.data[0]
            raise HTTPException(status_code=500, detail="Failed to mark notification as read")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def create_notification(data: NotificationCreate):
        try:
            notification_dict = data.dict()
            response = supabase_client.table(NOTIFICATIONS_TABLE).insert(notification_dict).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to create notification")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def delete_notification(notification_id: int):
        try:
            check = supabase_client.table(NOTIFICATIONS_TABLE).select("*").eq("id", notification_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Notification not found")
            
            supabase_client.table(NOTIFICATIONS_TABLE).delete().eq("id", notification_id).execute()
            return True
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

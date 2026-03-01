from fastapi import HTTPException
from API.core.supabase_client import supabase_client, USERS_TABLE
from API.schemas.profile_schemas import ProfileUpdate

class ProfileService:
    @staticmethod
    async def get_profile(user_id: int):
        try:
            response = supabase_client.table(USERS_TABLE).select("*").eq("id", user_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=404, detail="User not found")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_profile(user_id: int, data: ProfileUpdate):
        try:
            check = supabase_client.table(USERS_TABLE).select("*").eq("id", user_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="User not found")
            
            update_data = data.dict(exclude_unset=True)
            response = supabase_client.table(USERS_TABLE).update(update_data).eq("id", user_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to update profile")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

from fastapi import HTTPException
from API.schemas.match_schemas import MatchCreate, MatchUpdate
from API.core.supabase_client import supabase_client, MATCHES_TABLE

class MatchService:
    @staticmethod
    async def create_match(data: MatchCreate):
        try:
            match_dict = data.dict()
            response = supabase_client.table(MATCHES_TABLE).insert(match_dict).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to create match")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_matches():
        try:
            response = supabase_client.table(MATCHES_TABLE).select("*").execute()
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_match(match_id: int):
        try:
            response = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=404, detail="Match not found")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_match(match_id: int, data: MatchUpdate):
        try:
            check = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Match not found")
            
            update_data = data.dict(exclude_unset=True)
            response = supabase_client.table(MATCHES_TABLE).update(update_data).eq("id", match_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to update match")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def delete_match(match_id: int):
        try:
            check = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Match not found")
            
            supabase_client.table(MATCHES_TABLE).delete().eq("id", match_id).execute()
            return True
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

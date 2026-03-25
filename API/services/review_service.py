from fastapi import HTTPException
from API.schemas.review_schemas import ReviewCreate, ReviewUpdate
from API.core.supabase_client import supabase_client, REVIEWS_TABLE

class ReviewService:
    @staticmethod
    async def create_review(data: ReviewCreate):
        try:
            review_dict = data.dict()
            response = supabase_client.table(REVIEWS_TABLE).insert(review_dict).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to create review")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_reviews():
        try:
            response = supabase_client.table(REVIEWS_TABLE).select("*").execute()
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_review(review_id: int):
        try:
            response = supabase_client.table(REVIEWS_TABLE).select("*").eq("id", review_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=404, detail="Review not found")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_review(review_id: int, data: ReviewUpdate):
        try:
            check = supabase_client.table(REVIEWS_TABLE).select("*").eq("id", review_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Review not found")
            
            update_data = data.dict(exclude_unset=True)
            response = supabase_client.table(REVIEWS_TABLE).update(update_data).eq("id", review_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to update review")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def delete_review(review_id: int):
        try:
            check = supabase_client.table(REVIEWS_TABLE).select("*").eq("id", review_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Review not found")
            
            supabase_client.table(REVIEWS_TABLE).delete().eq("id", review_id).execute()
            return True
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

from fastapi import HTTPException
import logging
from API.schemas.review_schemas import ReviewCreate, ReviewUpdate
from API.core.supabase_client import supabase_client, REVIEWS_TABLE

logger = logging.getLogger("fairplay.api.reviews")

class ReviewService:
    @staticmethod
    async def create_review(data: ReviewCreate, user_id: int):
        try:
            review_dict = data.dict(exclude_unset=True)
            review_dict["user_id"] = user_id
            if "delivery" in review_dict and "over" not in review_dict:
                review_dict["over"] = review_dict["delivery"]
            review_dict.pop("delivery", None)
            logger.info("[create_review] Creating review for match_id=%s", review_dict.get("match_id"))
            if not review_dict.get("match_name"):
                match = supabase_client.table("matches").select("name").eq("id", review_dict["match_id"]).execute()
                if match.data:
                    review_dict["match_name"] = match.data[0]["name"]
                    logger.debug("[create_review] Resolved match_name=%r", review_dict["match_name"])

            response = supabase_client.table(REVIEWS_TABLE).insert(review_dict).execute()
            if response.data:
                logger.info("[create_review] Created review id=%s", response.data[0].get("id"))
                return response.data[0]
            logger.error("[create_review] Insert returned empty data | payload=%s", review_dict)
            raise HTTPException(status_code=500, detail="Failed to create review")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[create_review] ERROR | %s", e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_reviews(user_id: int):
        try:
            response = supabase_client.table(REVIEWS_TABLE).select("*").eq("user_id", user_id).execute()
            logger.info("[get_reviews] user_id=%s returned_rows=%d", user_id, len(response.data) if response.data else 0)
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_review(review_id: int, user_id: int):
        try:
            response = supabase_client.table(REVIEWS_TABLE).select("*").eq("id", review_id).eq("user_id", user_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=404, detail="Review not found")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_review(review_id: int, user_id: int, data: ReviewUpdate):
        try:
            check = supabase_client.table(REVIEWS_TABLE).select("*").eq("id", review_id).eq("user_id", user_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Review not found")
            
            update_data = data.dict(exclude_unset=True)
            if "delivery" in update_data and "over" not in update_data:
                update_data["over"] = update_data["delivery"]
            update_data.pop("delivery", None)
            response = supabase_client.table(REVIEWS_TABLE).update(update_data).eq("id", review_id).eq("user_id", user_id).execute()
            if response.data:
                return response.data[0]
            raise HTTPException(status_code=500, detail="Failed to update review")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_reviews_by_match(match_id: int, user_id: int):
        try:
            response = supabase_client.table(REVIEWS_TABLE).select("*").eq("match_id", match_id).eq("user_id", user_id).execute()
            logger.info("[get_reviews_by_match] user_id=%s match_id=%s returned_rows=%d", user_id, match_id, len(response.data) if response.data else 0)
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def delete_review(review_id: int, user_id: int):
        try:
            check = supabase_client.table(REVIEWS_TABLE).select("*").eq("id", review_id).eq("user_id", user_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Review not found")
            
            supabase_client.table(REVIEWS_TABLE).delete().eq("id", review_id).eq("user_id", user_id).execute()
            return True
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

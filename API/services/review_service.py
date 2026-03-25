from fastapi import HTTPException
import logging
from urllib.parse import urlparse, unquote
from API.schemas.review_schemas import ReviewCreate, ReviewUpdate
from API.core.supabase_client import supabase_client, supabase_admin_client, REVIEWS_TABLE

REVIEW_VIDEOS_BUCKET = "review-videos"
SIGNED_URL_EXPIRES_IN_SECONDS = 60 * 60  # 1 hour

logger = logging.getLogger("fairplay.api.reviews")

class ReviewService:
    _SELECT_FIELDS = (
        "id, match_id, user_id, match_name, over, original_decision, decision, impact, pitch, wickets, "
        "video_uri, content, analysis, created_at, updated_at"
    )

    @staticmethod
    def _write_client():
        return supabase_admin_client or supabase_client

    @staticmethod
    def _resolve_signed_url(signed_url_result) -> str:
        """
        supabase-py has returned various shapes over time. Normalize to a plain URL string.
        Expected typical shape: {"signedURL": "..."} or {"signedUrl": "..."} or {"signed_url": "..."}
        """
        if isinstance(signed_url_result, str):
            return signed_url_result
        if isinstance(signed_url_result, dict):
            return (
                signed_url_result.get("signedURL")
                or signed_url_result.get("signedUrl")
                or signed_url_result.get("signed_url")
                or signed_url_result.get("signed_url_url")  # defensive
                or ""
            )
        return ""

    @staticmethod
    def _extract_object_path_from_storage_url(url: str) -> str | None:
        """
        Convert a Supabase Storage URL into the object path expected by `create_signed_url`.

        Examples:
          https://<project>.supabase.co/storage/v1/object/public/review-videos/<object_path>
            -> <object_path>
          https://<project>.supabase.co/storage/v1/object/sign/review-videos/<object_path>?token=...
            -> <object_path>
        """
        try:
            parsed = urlparse(url)
            path = parsed.path or ""
        except Exception:
            return None

        public_prefix = f"/storage/v1/object/public/{REVIEW_VIDEOS_BUCKET}/"
        sign_prefix = f"/storage/v1/object/sign/{REVIEW_VIDEOS_BUCKET}/"

        if path.startswith(public_prefix):
            return unquote(path[len(public_prefix) :])
        if path.startswith(sign_prefix):
            return unquote(path[len(sign_prefix) :])
        return None

    @staticmethod
    def _attach_signed_video_url(review: dict) -> dict:
        """
        For private buckets: keep `video_uri` as the object path, and add `video_url` as a signed URL.
        If `video_uri` already looks like a URL, `video_url` will mirror it.
        """
        if not isinstance(review, dict):
            return review

        raw_video_uri = (review.get("video_uri") or "").strip()
        if not raw_video_uri:
            return review

        video_uri = raw_video_uri
        if raw_video_uri.lower().startswith("http://") or raw_video_uri.lower().startswith("https://"):
            extracted = ReviewService._extract_object_path_from_storage_url(raw_video_uri)
            if extracted:
                # Convert legacy stored public URL into object path so we can sign it (works for private buckets too).
                video_uri = extracted
            else:
                # Unknown URL shape; pass through.
                review["video_url"] = raw_video_uri
                return review

        try:
            storage = ReviewService._write_client().storage.from_(REVIEW_VIDEOS_BUCKET)
            signed = storage.create_signed_url(video_uri, SIGNED_URL_EXPIRES_IN_SECONDS)
            review["video_url"] = ReviewService._resolve_signed_url(signed)
        except Exception as exc:
            logger.warning("[signed_video_url] Failed for video_uri=%r | %s", video_uri, exc)
        return review

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
            response = (
                supabase_client.table(REVIEWS_TABLE)
                .select(ReviewService._SELECT_FIELDS)
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            logger.info("[get_reviews] user_id=%s returned_rows=%d", user_id, len(response.data) if response.data else 0)
            rows = response.data or []
            return [ReviewService._attach_signed_video_url(r) for r in rows]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_review(review_id: int, user_id: int):
        try:
            response = (
                supabase_client.table(REVIEWS_TABLE)
                .select(ReviewService._SELECT_FIELDS)
                .eq("id", review_id)
                .eq("user_id", user_id)
                .execute()
            )
            if response.data:
                return ReviewService._attach_signed_video_url(response.data[0])
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
            response = (
                supabase_client.table(REVIEWS_TABLE)
                .select(ReviewService._SELECT_FIELDS)
                .eq("match_id", match_id)
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            logger.info("[get_reviews_by_match] user_id=%s match_id=%s returned_rows=%d", user_id, match_id, len(response.data) if response.data else 0)
            rows = response.data or []
            return [ReviewService._attach_signed_video_url(r) for r in rows]
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

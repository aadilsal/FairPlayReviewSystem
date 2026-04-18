from fastapi import APIRouter, Depends
import logging
from API.schemas.review_schemas import ReviewCreate, ReviewUpdate, ReviewOut
from API.services.review_service import ReviewService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()
logger = logging.getLogger("fairplay.api.reviews")

@router.post("", response_model=dict)
@router.post("/", response_model=dict, include_in_schema=False)
async def create_review(review: ReviewCreate, current_user=Depends(get_current_user)):
    logger.info("Create review request by user_id=%s", current_user["id"])
    new_review = await ReviewService.create_review(review, current_user["id"])
    logger.info("Review created with id=%s", new_review.get("id") if isinstance(new_review, dict) else "unknown")
    return success_response(data=new_review, message="Review created")

@router.get("", response_model=dict)
@router.get("/", response_model=dict, include_in_schema=False)
async def get_reviews(current_user=Depends(get_current_user)):
    logger.info("Get reviews request received by user_id=%s", current_user["id"])
    reviews = await ReviewService.get_reviews(current_user["id"])
    logger.info("Get reviews request completed")
    return success_response(data=reviews)

@router.get("/{review_id}", response_model=dict)
async def get_review(review_id: int, current_user=Depends(get_current_user)):
    logger.info("Get review request for review_id=%s by user_id=%s", review_id, current_user["id"])
    review = await ReviewService.get_review(review_id, current_user["id"])
    return success_response(data=review)

@router.get("/match/{match_id}", response_model=dict)
async def get_reviews_by_match(match_id: int, current_user=Depends(get_current_user)):
    logger.info("Get reviews request for match_id=%s by user_id=%s", match_id, current_user["id"])
    reviews = await ReviewService.get_reviews_by_match(match_id, current_user["id"])
    return success_response(data=reviews)

@router.put("/{review_id}", response_model=dict)
async def update_review(review_id: int, review: ReviewUpdate, current_user=Depends(get_current_user)):
    logger.info("Update review request for review_id=%s by user_id=%s", review_id, current_user["id"])
    updated = await ReviewService.update_review(review_id, current_user["id"], review)
    logger.info("Review updated for review_id=%s", review_id)
    return success_response(data=updated, message="Review updated")

@router.delete("/{review_id}")
async def delete_review(review_id: int, current_user=Depends(get_current_user)):
    logger.info("Delete review request for review_id=%s by user_id=%s", review_id, current_user["id"])
    await ReviewService.delete_review(review_id, current_user["id"])
    logger.info("Review deleted for review_id=%s", review_id)
    return success_response(message="Review deleted")

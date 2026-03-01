from fastapi import APIRouter, Depends
from API.schemas.review_schemas import ReviewCreate, ReviewUpdate, ReviewOut
from API.services.review_service import ReviewService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()

@router.post("/", response_model=dict)
async def create_review(review: ReviewCreate, current_user=Depends(get_current_user)):
    new_review = await ReviewService.create_review(review)
    return success_response(data=new_review, message="Review created")

@router.get("/", response_model=dict)
async def get_reviews():
    reviews = await ReviewService.get_reviews()
    return success_response(data=reviews)

@router.get("/{review_id}", response_model=dict)
async def get_review(review_id: int):
    review = await ReviewService.get_review(review_id)
    return success_response(data=review)

@router.put("/{review_id}", response_model=dict)
async def update_review(review_id: int, review: ReviewUpdate, current_user=Depends(get_current_user)):
    updated = await ReviewService.update_review(review_id, review)
    return success_response(data=updated, message="Review updated")

@router.delete("/{review_id}")
async def delete_review(review_id: int, current_user=Depends(get_current_user)):
    await ReviewService.delete_review(review_id)
    return success_response(message="Review deleted")

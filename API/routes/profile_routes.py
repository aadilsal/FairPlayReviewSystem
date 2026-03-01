from fastapi import APIRouter, Depends
from API.services.profile_service import ProfileService
from API.schemas.profile_schemas import ProfileUpdate, ProfileOut
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()

@router.get("/", response_model=dict)
async def get_profile(current_user=Depends(get_current_user)):
    profile = await ProfileService.get_profile(current_user["id"])
    return success_response(data=profile)

@router.put("/", response_model=dict)
async def update_profile(update: ProfileUpdate, current_user=Depends(get_current_user)):
    updated = await ProfileService.update_profile(current_user["id"], update)
    return success_response(data=updated, message="Profile updated")

@router.post("/avatar")
async def upload_avatar():
    # Implement avatar upload logic
    return success_response(message="Avatar uploaded")

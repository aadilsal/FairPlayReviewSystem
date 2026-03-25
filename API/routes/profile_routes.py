from fastapi import APIRouter, Depends, File, UploadFile
import logging
from API.services.profile_service import ProfileService
from API.schemas.profile_schemas import ProfileUpdate, ProfileOut
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()
logger = logging.getLogger("fairplay.api.profile")

@router.get("", response_model=dict)
@router.get("/", response_model=dict, include_in_schema=False)
async def get_profile(current_user=Depends(get_current_user)):
    logger.info("Profile fetch request for user_id=%s", current_user["id"])
    profile = await ProfileService.get_profile(current_user["id"])
    return success_response(data=profile)

@router.put("", response_model=dict)
@router.put("/", response_model=dict, include_in_schema=False)
async def update_profile(update: ProfileUpdate, current_user=Depends(get_current_user)):
    logger.info("Profile update request for user_id=%s", current_user["id"])
    updated = await ProfileService.update_profile(current_user["id"], update)
    logger.info("Profile update completed for user_id=%s", current_user["id"])
    return success_response(data=updated, message="Profile updated")

@router.post("/avatar")
async def upload_avatar(avatar: UploadFile = File(...), current_user=Depends(get_current_user)):
    logger.info("Avatar upload endpoint called for user_id=%s", current_user["id"])
    updated = await ProfileService.upload_avatar_file(current_user["id"], avatar)
    return success_response(data=updated, message="Avatar uploaded")

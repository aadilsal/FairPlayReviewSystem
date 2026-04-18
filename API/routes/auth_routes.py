from fastapi import APIRouter, Depends, status, HTTPException
import logging
from API.schemas.auth_schemas import UserCreate, UserLogin, ChangePassword, UserOut, Token
from API.services.auth_service import AuthService
from API.core.security import create_access_token
from API.utils.response_formatter import success_response, error_response
from API.dependencies.auth_dependency import get_current_user, get_current_user_optional

router = APIRouter()
logger = logging.getLogger("fairplay.api.auth")

@router.post("/signup", response_model=dict)
async def signup(user: UserCreate):
    logger.info("Signup request received for email=%s", user.email)
    new_user = await AuthService.signup(user)
    logger.info("Signup successful for email=%s", user.email)
    # The frontend expects {user: {...}, access_token: "...", token_type: "bearer"} on signup too? 
    # Let's check api.types.ts: "Backend returns user + token on both login and signup"
    access_token = create_access_token({"sub": str(new_user["id"])})
    response = success_response(data={
        "user": new_user,
        "access_token": access_token,
        "token_type": "bearer"
    }, message="Signup successful")
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return response

@router.post("/login", response_model=dict)
async def login(user: UserLogin):
    logger.info("Login request received for email=%s", user.email)
    token, user_obj = await AuthService.login(user)
    logger.info("Login successful for email=%s", user.email)
    response = success_response(data={
        "user": user_obj,
        "access_token": token, 
        "token_type": "bearer"
    }, message="Login successful")
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return response

@router.post("/change-password")
async def change_password(data: ChangePassword, current_user=Depends(get_current_user_optional)):
    if current_user is not None:
        logger.info("Change password request for user_id=%s", current_user["id"])
        await AuthService.change_password(current_user["id"], data)
        logger.info("Password changed for user_id=%s", current_user["id"])
        return success_response(message="Password changed successfully")

    logger.warning("Change password called without auth token; attempting email+old_password fallback")
    await AuthService.change_password_by_email(data)
    logger.info("Password changed via email fallback for email=%s", data.email)
    return success_response(message="Password changed successfully")

@router.get("/profile", response_model=dict)
async def get_profile(current_user=Depends(get_current_user)):
    logger.info("Profile fetch for user_id=%s", current_user["id"])
    # current_user from dependency might still have "username"
    if "username" in current_user:
        current_user["name"] = current_user.pop("username")
    return success_response(data=current_user)

@router.put("/profile", response_model=dict)
async def update_profile(update: dict, current_user=Depends(get_current_user)):
    logger.info("Profile update request for user_id=%s", current_user["id"])
    updated_user = await AuthService.update_profile(current_user["id"], update)
    logger.info("Profile update successful for user_id=%s", current_user["id"])
    return success_response(data=updated_user, message="Profile updated")

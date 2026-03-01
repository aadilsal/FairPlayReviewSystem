from fastapi import APIRouter
from API.utils.response_formatter import success_response

router = APIRouter()

@router.get("/")
async def health_check():
    return success_response(message="API is healthy")

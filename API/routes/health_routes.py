from fastapi import APIRouter
import logging
from API.utils.response_formatter import success_response

router = APIRouter()
logger = logging.getLogger("fairplay.api.health")

@router.get("/")
async def health_check():
    logger.info("Health check endpoint called")
    return success_response(message="API is healthy")

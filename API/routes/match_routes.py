from fastapi import APIRouter, Depends, Query
import logging
from API.schemas.match_schemas import MatchCreate, MatchUpdate, MatchOut
from API.services.match_service import MatchService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()
logger = logging.getLogger("fairplay.api.matches")

@router.post(
    "",
    response_model=dict,
    summary="Create match",
    description="Create a user-owned match. Status aliases like upcoming/live are normalized to DB-safe values.",
)
@router.post("/", response_model=dict, include_in_schema=False)
async def create_match(match: MatchCreate, current_user=Depends(get_current_user)):
    logger.info("Create match request by user_id=%s", current_user["id"])
    new_match = await MatchService.create_match(match, current_user["id"])
    logger.info("Match created with id=%s", new_match.get("id") if isinstance(new_match, dict) else "unknown")
    return success_response(data=new_match, message="Match created")

@router.get(
    "",
    response_model=dict,
    summary="List matches",
    description="List current user's matches. Triggers stale in-progress auto-complete check before returning rows.",
)
@router.get("/", response_model=dict, include_in_schema=False)
async def get_matches(current_user=Depends(get_current_user)):
    logger.info("Get matches request received by user_id=%s", current_user["id"])
    matches = await MatchService.get_matches(current_user["id"])
    logger.info("Get matches request completed")
    return success_response(data=matches)

@router.get(
    "/{match_id}",
    response_model=dict,
    summary="Get match",
    description="Get one match by ID for the current user. Also triggers stale auto-complete check.",
)
async def get_match(match_id: int, current_user=Depends(get_current_user)):
    logger.info("Get match request for match_id=%s by user_id=%s", match_id, current_user["id"])
    match = await MatchService.get_match(match_id, current_user["id"])
    return success_response(data=match)

@router.put(
    "/{match_id}",
    response_model=dict,
    summary="Update match",
    description="Update match fields such as status, date, and teams. Setting status to completed works for manual completion.",
)
async def update_match(match_id: int, match: MatchUpdate, current_user=Depends(get_current_user)):
    logger.info("Update match request for match_id=%s by user_id=%s", match_id, current_user["id"])
    updated = await MatchService.update_match(match_id, current_user["id"], match)
    logger.info("Match updated for match_id=%s", match_id)
    return success_response(data=updated, message="Match updated")

@router.delete(
    "/{match_id}",
    summary="Delete match",
    description="Delete a match owned by the current user.",
)
async def delete_match(match_id: int, current_user=Depends(get_current_user)):
    logger.info("Delete match request for match_id=%s by user_id=%s", match_id, current_user["id"])
    await MatchService.delete_match(match_id, current_user["id"])
    logger.info("Match deleted for match_id=%s", match_id)
    return success_response(message="Match deleted")


@router.post(
    "/maintenance/auto-complete",
    response_model=dict,
    summary="Run stale match auto-complete",
    description="Manually trigger auto-completion for in-progress matches inactive for timeout_hours (default 24).",
)
async def auto_complete_stale_matches(
    timeout_hours: int = Query(default=24, ge=1, le=168),
    current_user=Depends(get_current_user),
):
    logger.info(
        "Auto-complete maintenance request by user_id=%s timeout_hours=%s",
        current_user["id"],
        timeout_hours,
    )
    result = await MatchService.auto_complete_stale_matches(timeout_hours, user_id=current_user["id"])
    return success_response(data=result, message="Stale in-progress matches auto-completed")


@router.post(
    "/{match_id}/heartbeat",
    response_model=dict,
    summary="Heartbeat active match",
    description="Refresh updated_at for an in-progress match to prevent stale auto-complete while user is active.",
)
async def heartbeat_match(match_id: int, current_user=Depends(get_current_user)):
    logger.info("Match heartbeat for match_id=%s by user_id=%s", match_id, current_user["id"])
    updated = await MatchService.touch_match_activity(match_id, current_user["id"])
    return success_response(data=updated, message="Match activity heartbeat recorded")

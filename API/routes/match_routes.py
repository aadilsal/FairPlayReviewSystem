from fastapi import APIRouter, BackgroundTasks, Depends, Query
import logging
from API.schemas.match_schemas import MatchCreate, MatchUpdate, MatchOut
from API.services.match_service import MatchService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user
from fastapi import UploadFile, File
from API.schemas.wicket_schemas import WicketConfigurationManualUpdate
from API.services.wicket_config_service import WicketConfigService
from API.utils.response_formatter import error_response
from API.utils.file_handler import save_upload_image_file

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


@router.get(
    "/{match_id}/wicket-config",
    response_model=dict,
    summary="Get wicket configuration",
    description="Return saved wicket coordinates (near/far) for the match for this user.",
)
async def get_wicket_config(match_id: int, current_user=Depends(get_current_user)):
    cfg = WicketConfigService.get_config(match_id, current_user["id"])
    data = {
        "match_id": match_id,
        "user_id": current_user["id"],
        "configured": bool(cfg.get("configured")) if cfg else False,
        "status": cfg.get("status") if cfg else "idle",
        "near_box": cfg.get("near_box") if cfg else None,
        "far_box": cfg.get("far_box") if cfg else None,
        "error_message": cfg.get("error_message") if cfg else None,
        "annotated_image_path": cfg.get("annotated_image_path") if cfg else None,
        "annotated_image_object_path": cfg.get("annotated_image_object_path") if cfg else None,
        "source_image_object_path": cfg.get("source_image_object_path") if cfg else None,
        "updated_at": cfg.get("updated_at") if cfg else None,
    }
    return success_response(data=data, message="Wicket configuration fetched")


@router.post(
    "/{match_id}/wicket-config/auto",
    response_model=dict,
    summary="Auto-detect and save wicket configuration",
    description="Runs wicket detection once on an uploaded image and persists near/far wicket boxes for this match.",
)
async def auto_configure_wicket(
    match_id: int,
    background_tasks: BackgroundTasks,
    image_file: UploadFile | None = File(default=None),
    # Backward compatibility: some clients still send `video_file`.
    video_file: UploadFile | None = File(default=None),
    wicket_conf: float = 0.25,
    display: bool = True,
    current_user=Depends(get_current_user),
):
    logger.info(
        "Auto wicket-config request match_id=%s user_id=%s image_file=%s video_file=%s wicket_conf=%s display=%s",
        match_id,
        current_user["id"],
        getattr(image_file, "filename", None),
        getattr(video_file, "filename", None),
        wicket_conf,
        display,
    )

    chosen = image_file or video_file
    if chosen is None:
        logger.warning(
            "Auto wicket-config missing file field match_id=%s user_id=%s",
            match_id,
            current_user["id"],
        )
        return error_response(
            message="Missing file. Send multipart/form-data with `image_file` (preferred) or `video_file` (legacy).",
            status_code=422,
        )

    # Persist upload to disk immediately, then return right away and run detection in background.
    try:
        image_path = save_upload_image_file(chosen)
    except Exception as exc:
        logger.exception("Failed to save wicket-config image match_id=%s user_id=%s", match_id, current_user["id"])
        raise

    # Mark DB row as processing (so frontend can poll GET /wicket-config).
    row = WicketConfigService.mark_processing(match_id, current_user["id"])

    def _run_background():
        try:
            logger.info(
                "Background wicket auto-config START match_id=%s user_id=%s path=%s",
                match_id,
                current_user["id"],
                image_path,
            )
            WicketConfigService.auto_configure_from_image_path(
                match_id=match_id,
                user_id=current_user["id"],
                image_path=image_path,
                wicket_conf=wicket_conf,
                display=display,
            )
            logger.info("Background wicket auto-config DONE match_id=%s user_id=%s", match_id, current_user["id"])
        except Exception as exc:
            logger.exception(
                "Background wicket auto-config FAILED match_id=%s user_id=%s error=%s",
                match_id,
                current_user["id"],
                str(exc),
            )
            try:
                WicketConfigService.mark_failed(match_id, current_user["id"], str(exc))
            except Exception:
                logger.exception("Failed to mark wicket-config as failed match_id=%s user_id=%s", match_id, current_user["id"])

    if background_tasks is not None:
        background_tasks.add_task(_run_background)
    else:
        # Fallback (shouldn't happen) - run inline.
        _run_background()

    data = {
        "match_id": match_id,
        "user_id": current_user["id"],
        "configured": bool(row.get("configured") or False),
        "status": row.get("status") or "processing",
        "near_box": row.get("near_box"),
        "far_box": row.get("far_box"),
        "error_message": row.get("error_message"),
        "updated_at": row.get("updated_at"),
    }
    return success_response(data=data, message="Wicket auto-configuration started")


@router.put(
    "/{match_id}/wicket-config",
    response_model=dict,
    summary="Manually update wicket configuration",
    description="Overrides wicket coordinates (near/far) for the match and marks it as configured.",
)
async def update_wicket_config(
    match_id: int,
    payload: WicketConfigurationManualUpdate,
    current_user=Depends(get_current_user),
):
    row = WicketConfigService.upsert_manual_config(
        match_id=match_id,
        user_id=current_user["id"],
        near_box=payload.near_box.as_list(),
        far_box=payload.far_box.as_list(),
        configured=bool(payload.configured),
    )
    data = {
        "match_id": match_id,
        "user_id": current_user["id"],
        "configured": bool(row.get("configured")),
        "near_box": row.get("near_box"),
        "far_box": row.get("far_box"),
        "updated_at": row.get("updated_at"),
    }
    return success_response(data=data, message="Wicket configuration updated")

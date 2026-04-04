from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse

from API.core.supabase_client import supabase_admin_client, supabase_client

router = APIRouter()
logger = logging.getLogger("fairplay.api.video_proxy")

REVIEW_VIDEOS_BUCKET = "review-videos"
SIGNED_URL_EXPIRES_IN_SECONDS = 60 * 60  # 1 hour


def _write_client():
    return supabase_admin_client or supabase_client


def _resolve_signed_url(signed_url_result) -> str:
    if isinstance(signed_url_result, str):
        return signed_url_result
    if isinstance(signed_url_result, dict):
        return signed_url_result.get("signedURL") or signed_url_result.get("signedUrl") or ""
    return ""


@router.get("/reviews/{object_path:path}")
async def get_review_video(object_path: str):
    """
    Backend-only compatibility endpoint.

    Frontend may request /reviews/<object_path> for videos (.mp4) or LBW card images (.jpg).
    Objects live in Supabase Storage (private bucket); we return a short-lived signed URL via redirect.
    """
    safe = (object_path or "").strip().lstrip("/")
    if not safe:
        raise HTTPException(status_code=400, detail="Missing object path")

    # Frontend sends: /reviews/user_5/match_15/<file>.mp4
    # Storage keys are: reviews/user_5/match_15/<file>.mp4
    storage_key = safe if safe.startswith("reviews/") else f"reviews/{safe}"

    try:
        storage = _write_client().storage.from_(REVIEW_VIDEOS_BUCKET)
        signed = storage.create_signed_url(storage_key, SIGNED_URL_EXPIRES_IN_SECONDS)
        url = _resolve_signed_url(signed)
    except Exception as exc:
        logger.exception("Failed to sign video url for key=%r", storage_key)
        raise HTTPException(status_code=500, detail=f"Failed to generate video URL: {exc}")

    if not url:
        raise HTTPException(status_code=404, detail="Video not found")

    # Redirect lets the browser stream directly from Supabase.
    return RedirectResponse(url=url, status_code=307)


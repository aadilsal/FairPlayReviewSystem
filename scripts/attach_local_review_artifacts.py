from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Allow running as a script from `scripts/` (ensures `import API...` works).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from API.core.supabase_client import (
    DETECTION_RESULTS_TABLE,
    MATCHES_TABLE,
    NOTIFICATIONS_TABLE,
    REVIEWS_TABLE,
    supabase_admin_client,
    supabase_client,
)


REVIEW_VIDEOS_BUCKET = "review-videos"


@dataclass(frozen=True)
class UploadedArtifacts:
    video_object_path: str
    summary_object_path: str


def _write_client():
    # Prefer service-role for Storage uploads when available.
    return supabase_admin_client or supabase_client


def _require_env() -> None:
    # create_client() was already called at import time; this is a friendlier error for missing config.
    url = os.getenv("SUPABASE_URL", "") or ""
    key = os.getenv("SUPABASE_KEY", "") or ""
    if not url.strip() or not key.strip():
        raise RuntimeError(
            "Missing SUPABASE_URL / SUPABASE_KEY. Add them to an .env file in the repo root (or set environment variables) "
            "then rerun the script."
        )
    # Storage uploads often fail with RLS unless service-role is used.
    service = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or ""
    if not service.strip():
        raise RuntimeError(
            "Missing SUPABASE_SERVICE_ROLE_KEY. Storage uploads to bucket 'review-videos' are blocked by RLS with anon keys. "
            "Add SUPABASE_SERVICE_ROLE_KEY to your .env and rerun."
        )


def _upload_bytes(*, object_path: str, payload: bytes, content_type: str) -> None:
    if not payload:
        raise RuntimeError("Refusing to upload empty payload")
    storage = _write_client().storage.from_(REVIEW_VIDEOS_BUCKET)
    try:
        storage.upload(object_path, payload, {"content-type": content_type, "upsert": "true"})
    except TypeError:
        # Older supabase-py versions accept fewer args.
        storage.upload(object_path, payload)


def _upload_artifacts(
    *,
    user_id: int,
    match_id: int,
    local_video: Path,
    local_summary_json: Path,
) -> UploadedArtifacts:
    if not local_video.exists():
        raise FileNotFoundError(str(local_video))
    if not local_summary_json.exists():
        raise FileNotFoundError(str(local_summary_json))

    video_payload = local_video.read_bytes()
    summary_payload = local_summary_json.read_bytes()

    base_dir = f"reviews/user_{user_id}/match_{match_id}"
    video_object_path = f"{base_dir}/t4_output_{uuid.uuid4().hex}.mp4"
    summary_object_path = f"{base_dir}/lbw_summary_{uuid.uuid4().hex}.json"

    _upload_bytes(object_path=video_object_path, payload=video_payload, content_type="video/mp4")
    _upload_bytes(object_path=summary_object_path, payload=summary_payload, content_type="application/json")

    return UploadedArtifacts(video_object_path=video_object_path, summary_object_path=summary_object_path)


def _get_match_name(*, match_id: int, user_id: int) -> Optional[str]:
    resp = (
        supabase_client.table(MATCHES_TABLE)
        .select("id, name, teams")
        .eq("id", match_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [None])[0]
    if not isinstance(row, dict):
        return None
    return (row.get("name") or row.get("teams") or f"Match {match_id}").strip()


def _get_match_row(*, match_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    resp = (
        supabase_client.table(MATCHES_TABLE)
        .select("*")
        .eq("id", match_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [None])[0]
    return row if isinstance(row, dict) else None


def _ensure_match_fields(*, match_id: int, user_id: int) -> Dict[str, Any]:
    """
    Ensure match is "frontend-complete" where possible (name/teams).
    We only backfill from existing columns; we do not invent venue/date.
    """
    row = _get_match_row(match_id=match_id, user_id=user_id)
    if not row:
        raise RuntimeError(f"Match not found or not owned by user_id={user_id}: match_id={match_id}")

    team_a = (row.get("team_a") or "").strip() if row.get("team_a") is not None else ""
    team_b = (row.get("team_b") or "").strip() if row.get("team_b") is not None else ""
    derived_teams = " vs ".join([p for p in [team_a, team_b] if p]) or None

    patch: Dict[str, Any] = {}
    if not (row.get("teams") or "").strip():
        if derived_teams:
            patch["teams"] = derived_teams
    if not (row.get("name") or "").strip():
        patch["name"] = patch.get("teams") or derived_teams or f"Match {match_id}"

    if patch:
        resp = _write_client().table(MATCHES_TABLE).update(patch).eq("id", match_id).execute()
        updated = (resp.data or [None])[0]
        if isinstance(updated, dict):
            row = updated
    return row


def _lbw_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "t", "1", "yes", "y"):
            return True
        if v in ("false", "f", "0", "no", "n"):
            return False
    return None


def _lbw_label(flag: Optional[bool], *, true_label: str, false_label: str) -> Optional[str]:
    if flag is True:
        return true_label
    if flag is False:
        return false_label
    return None


def _derive_review_fields_from_summary(summary_json: Dict[str, Any]) -> Dict[str, Any]:
    pitch_inline = _lbw_flag(summary_json.get("pitch_inline"))
    impact_inline = _lbw_flag(summary_json.get("impact_inline"))
    wickets_hitting = _lbw_flag(summary_json.get("wickets_hitting"))

    pitch = _lbw_label(pitch_inline, true_label="IN LINE", false_label="NOT IN LINE")
    impact = _lbw_label(impact_inline, true_label="IN LINE", false_label="NOT IN LINE")

    wickets: Optional[str] = None
    if wickets_hitting is True:
        wickets = "HITTING"
    elif wickets_hitting is False:
        wickets = "MISSING"

    decision: Optional[str] = None
    if pitch_inline is not None and impact_inline is not None and wickets_hitting is not None:
        decision = "OUT" if (pitch_inline and impact_inline and wickets_hitting) else "NOT OUT"

    return {"pitch": pitch, "impact": impact, "wickets": wickets, "decision": decision}


def _insert_review(
    *,
    match_id: int,
    user_id: int,
    match_name: Optional[str],
    video_object_path: str,
    content: str,
    analysis: Optional[str],
    original_decision: Optional[str],
    derived_fields: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "match_id": match_id,
        "user_id": user_id,
        "match_name": match_name,
        "video_uri": video_object_path,
        "content": (content or "").strip() or "Review submitted",
        "analysis": analysis,
        "original_decision": original_decision,
    }
    payload.update({k: v for k, v in (derived_fields or {}).items() if v is not None})
    resp = supabase_client.table(REVIEWS_TABLE).insert(payload).execute()
    if not resp.data:
        raise RuntimeError("Failed to insert review row (empty response)")
    return resp.data[0]


def _parse_summary_json(local_summary_json: Path) -> Tuple[Dict[str, Any], bytes]:
    raw_bytes = local_summary_json.read_bytes()
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON in {local_summary_json}: {exc}") from exc
    if not isinstance(parsed, dict):
        # DB column is JSONB; allow non-object, but normalize for predictable access.
        parsed = {"value": parsed}
    return parsed, raw_bytes


def _upsert_detection_result(
    *,
    match_id: int,
    user_id: int,
    output_video_object_path: str,
    summary_object_path: str,
    summary_json: Dict[str, Any],
    status: str = "completed",
) -> Dict[str, Any]:
    # Try to update the latest detection_results row for this match/user; otherwise insert a new one.
    existing = (
        supabase_client.table(DETECTION_RESULTS_TABLE)
        .select("id, created_at")
        .eq("match_id", match_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    row = (existing.data or [None])[0]
    patch: Dict[str, Any] = {
        "output_video_path": output_video_object_path,
        "metadata_path": summary_object_path,
        "summary_stats": summary_json,
        "result_data": summary_json,
        "status": status,
        "error_message": None,
    }
    if isinstance(row, dict) and row.get("id") is not None:
        resp = supabase_client.table(DETECTION_RESULTS_TABLE).update(patch).eq("id", row["id"]).execute()
        if not resp.data:
            raise RuntimeError("Failed to update detection_results row")
        return resp.data[0]

    payload = {
        "match_id": match_id,
        "user_id": user_id,
        "input_video_path": None,
        "output_video_path": output_video_object_path,
        "metadata_path": summary_object_path,
        "summary_stats": summary_json,
        "result_data": summary_json,
        "status": status,
        "processing_time_ms": None,
        "error_message": None,
    }
    resp = supabase_client.table(DETECTION_RESULTS_TABLE).insert(payload).execute()
    if not resp.data:
        raise RuntimeError("Failed to insert detection_results row")
    return resp.data[0]


def _insert_notification(*, user_id: int, message: str) -> None:
    msg = (message or "").strip()
    if not msg:
        return
    _write_client().table(NOTIFICATIONS_TABLE).insert({"user_id": user_id, "message": msg, "read": False}).execute()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a local MP4 + lbw_summary.json to Supabase Storage and attach them to reviews + detection_results."
    )
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--video", type=str, required=True, help="Local path to .mp4")
    parser.add_argument("--summary", type=str, required=True, help="Local path to lbw_summary.json")
    parser.add_argument("--content", type=str, default="LBW review video attached")
    parser.add_argument("--analysis", type=str, default=None)
    parser.add_argument("--original-decision", type=str, default=None, help="Optional: OUT / NOT OUT")
    args = parser.parse_args()

    # Ensure imports work when running from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    _require_env()

    user_id = int(args.user_id)
    match_id = int(args.match_id)
    local_video = Path(args.video)
    local_summary_json = Path(args.summary)

    summary_json, _ = _parse_summary_json(local_summary_json)
    derived_fields = _derive_review_fields_from_summary(summary_json)

    artifacts = _upload_artifacts(
        user_id=user_id,
        match_id=match_id,
        local_video=local_video,
        local_summary_json=local_summary_json,
    )

    match_row = _ensure_match_fields(match_id=match_id, user_id=user_id)
    match_name = (match_row.get("name") or match_row.get("teams") or f"Match {match_id}").strip()

    review = _insert_review(
        match_id=match_id,
        user_id=user_id,
        match_name=match_name,
        video_object_path=artifacts.video_object_path,
        content=args.content,
        analysis=args.analysis,
        original_decision=args.original_decision,
        derived_fields=derived_fields,
    )

    detection_result = _upsert_detection_result(
        match_id=match_id,
        user_id=user_id,
        output_video_object_path=artifacts.video_object_path,
        summary_object_path=artifacts.summary_object_path,
        summary_json=summary_json,
        status="completed",
    )

    _insert_notification(
        user_id=user_id,
        message=f"Review artifacts uploaded for {match_name} (match_id={match_id}).",
    )

    print("OK")
    print(f"review_id={review.get('id')} video_uri={review.get('video_uri')}")
    print(
        "detection_results:",
        f"id={detection_result.get('id')}",
        f"output_video_path={detection_result.get('output_video_path')}",
        f"metadata_path={detection_result.get('metadata_path')}",
    )


if __name__ == "__main__":
    main()


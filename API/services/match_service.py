from fastapi import HTTPException
import logging
from datetime import datetime
from dateutil import parser as date_parser
from API.schemas.match_schemas import MatchCreate, MatchUpdate
from API.core.supabase_client import supabase_client, supabase_admin_client, MATCHES_TABLE

logger = logging.getLogger("fairplay.api.matches")

DEFAULT_AUTO_COMPLETE_HOURS = 24

_STATUS_ALIASES = {
    "upcoming": "scheduled",
    "scheduled": "scheduled",
    "live": "in_progress",
    "in_progress": "in_progress",
    "in-progress": "in_progress",
    "ongoing": "in_progress",
    "completed": "completed",
    "finished": "completed",
    "done": "completed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "postponed": "postponed",
}


def _split_teams(teams: str) -> tuple[str, str]:
    """Parse 'Team A vs Team B' into team_a/team_b for legacy DB columns."""
    if not teams:
        return "", ""
    if " vs " in teams:
        a, b = teams.split(" vs ", 1)
        return a.strip(), b.strip()
    if " vs. " in teams:
        a, b = teams.split(" vs. ", 1)
        return a.strip(), b.strip()
    return teams.strip(), ""


def _normalize_match_date(value: str) -> str:
    """Accept flexible date strings from frontend and normalize to ISO datetime."""
    if not value:
        raise HTTPException(status_code=400, detail="date is required")

    cleaned = value.replace("\u202f", " ").strip()
    try:
        parsed = date_parser.parse(cleaned)
        return parsed.isoformat()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format")


def _normalize_match_status(value: str | None) -> str:
    """Map frontend status values to DB-safe values accepted by current constraint."""
    if not value:
        return "scheduled"
    normalized = _STATUS_ALIASES.get(str(value).strip().lower())
    if not normalized:
        raise HTTPException(status_code=400, detail=f"Invalid status value: {value}")
    return normalized


def _status_metadata_patch(status: str) -> dict:
    """Ensure completion metadata remains consistent with status transitions."""
    if status == "completed":
        return {
            "completed_by_system": False,
            "auto_completed_at": None,
            "completion_reason": None,
        }
    if status == "in_progress":
        return {
            "completed_by_system": False,
            "auto_completed_at": None,
            "completion_reason": None,
        }
    return {}


def _without_completion_metadata(payload: dict) -> dict:
    cleaned = dict(payload)
    cleaned.pop("completed_by_system", None)
    cleaned.pop("auto_completed_at", None)
    cleaned.pop("completion_reason", None)
    return cleaned


def _is_missing_column_error(exc: Exception, column_name: str) -> bool:
    text = str(exc)
    return "PGRST204" in text and f"'{column_name}'" in text


def _is_missing_rpc_error(exc: Exception, function_name: str) -> bool:
    text = str(exc)
    return function_name in text and ("PGRST202" in text or "not found" in text.lower())


def _rpc_client():
    """Prefer service-role client for maintenance RPC calls and cross-table writes."""
    return supabase_admin_client or supabase_client

class MatchService:
    @staticmethod
    async def create_match(data: MatchCreate, user_id: int):
        try:
            match_dict = data.dict()
            match_dict["user_id"] = user_id
            match_dict["date"] = _normalize_match_date(match_dict.get("date"))
            match_dict["status"] = _normalize_match_status(match_dict.get("status"))
            match_dict.update(_status_metadata_patch(match_dict["status"]))
            team_a, team_b = _split_teams(match_dict.get("teams", ""))
            match_dict["team_a"] = team_a
            match_dict["team_b"] = team_b
            logger.info("[create_match] Inserting match: %s", match_dict.get("teams") or match_dict)
            try:
                response = supabase_client.table(MATCHES_TABLE).insert(match_dict).execute()
            except Exception as e:
                # Migration-safe fallback: if completion metadata columns are not in DB yet,
                # retry insert without those fields.
                if _is_missing_column_error(e, "auto_completed_at") or _is_missing_column_error(e, "completed_by_system") or _is_missing_column_error(e, "completion_reason"):
                    logger.warning("[create_match] Completion metadata columns missing in DB, retrying insert without metadata fields")
                    fallback_dict = _without_completion_metadata(match_dict)
                    response = supabase_client.table(MATCHES_TABLE).insert(fallback_dict).execute()
                else:
                    raise
            if response.data:
                logger.info("[create_match] Created match id=%s", response.data[0].get("id"))
                return response.data[0]
            logger.error("[create_match] Insert returned empty data | payload=%s", match_dict)
            raise HTTPException(status_code=500, detail="Failed to create match")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[create_match] ERROR | %s", e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_matches(user_id: int):
        try:
            await MatchService.auto_complete_stale_matches(DEFAULT_AUTO_COMPLETE_HOURS, user_id=user_id)
            response = supabase_client.table(MATCHES_TABLE).select("*").eq("user_id", user_id).order("date", desc=True).execute()
            logger.info("[get_matches] user_id=%s returned_rows=%d", user_id, len(response.data) if response.data else 0)
            return response.data
        except Exception as e:
            logger.exception("[get_matches] ERROR | %s", e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_match(match_id: int, user_id: int):
        try:
            await MatchService.auto_complete_stale_matches(DEFAULT_AUTO_COMPLETE_HOURS, user_id=user_id)
            response = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).eq("user_id", user_id).execute()
            if response.data:
                return response.data[0]
            logger.warning("[get_match] Not found id=%s", match_id)
            raise HTTPException(status_code=404, detail="Match not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[get_match] ERROR match_id=%s | %s", match_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_match(match_id: int, user_id: int, data: MatchUpdate):
        try:
            check = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).eq("user_id", user_id).execute()
            if not check.data:
                logger.warning("[update_match] Not found id=%s", match_id)
                raise HTTPException(status_code=404, detail="Match not found")

            update_data = data.dict(exclude_unset=True)
            if "date" in update_data:
                update_data["date"] = _normalize_match_date(update_data.get("date"))
            if "status" in update_data:
                update_data["status"] = _normalize_match_status(update_data.get("status"))
                update_data.update(_status_metadata_patch(update_data["status"]))
            if "teams" in update_data:
                team_a, team_b = _split_teams(update_data.get("teams", ""))
                update_data["team_a"] = team_a
                update_data["team_b"] = team_b

            logger.debug("[update_match] Updating id=%s fields=%s", match_id, list(update_data.keys()))
            try:
                response = supabase_client.table(MATCHES_TABLE).update(update_data).eq("id", match_id).eq("user_id", user_id).execute()
            except Exception as e:
                # Migration-safe fallback: if completion metadata columns are not in DB yet,
                # retry update without those fields so manual updates keep working.
                if _is_missing_column_error(e, "auto_completed_at") or _is_missing_column_error(e, "completed_by_system") or _is_missing_column_error(e, "completion_reason"):
                    logger.warning("[update_match] Completion metadata columns missing in DB, retrying update without metadata fields")
                    fallback_update = _without_completion_metadata(update_data)
                    response = supabase_client.table(MATCHES_TABLE).update(fallback_update).eq("id", match_id).eq("user_id", user_id).execute()
                else:
                    raise
            if response.data:
                logger.info("[update_match] Updated id=%s", match_id)
                return response.data[0]
            logger.error("[update_match] Update returned empty data for id=%s", match_id)
            raise HTTPException(status_code=500, detail="Failed to update match")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[update_match] ERROR match_id=%s | %s", match_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def delete_match(match_id: int, user_id: int):
        try:
            check = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).eq("user_id", user_id).execute()
            if not check.data:
                logger.warning("[delete_match] Not found id=%s", match_id)
                raise HTTPException(status_code=404, detail="Match not found")
            supabase_client.table(MATCHES_TABLE).delete().eq("id", match_id).eq("user_id", user_id).execute()
            logger.info("[delete_match] Deleted id=%s", match_id)
            return True
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[delete_match] ERROR match_id=%s | %s", match_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def touch_match_activity(match_id: int, user_id: int):
        """Heartbeat endpoint for active clients to prevent stale auto-complete."""
        try:
            check = supabase_client.table(MATCHES_TABLE).select("*").eq("id", match_id).eq("user_id", user_id).execute()
            if not check.data:
                logger.warning("[touch_match_activity] Not found id=%s user_id=%s", match_id, user_id)
                raise HTTPException(status_code=404, detail="Match not found")

            match_row = check.data[0]
            if match_row.get("status") != "in_progress":
                raise HTTPException(status_code=400, detail="Heartbeat is only valid for in_progress matches")

            response = supabase_client.table(MATCHES_TABLE).update({"updated_at": datetime.utcnow().isoformat()}).eq("id", match_id).eq("user_id", user_id).execute()
            if response.data:
                logger.info("[touch_match_activity] Heartbeat recorded for match_id=%s user_id=%s", match_id, user_id)
                return response.data[0]

            raise HTTPException(status_code=500, detail="Failed to record match heartbeat")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[touch_match_activity] ERROR match_id=%s user_id=%s | %s", match_id, user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def auto_complete_stale_matches(timeout_hours: int = DEFAULT_AUTO_COMPLETE_HOURS, user_id: int | None = None):
        """Auto-complete stale in_progress matches and send notifications."""
        try:
            safe_hours = max(int(timeout_hours), 1)
            rpc_params = {
                "timeout_hours": safe_hours,
                "target_user_id": user_id,
            }
            client = _rpc_client()
            response = client.rpc("auto_complete_stale_matches", rpc_params).execute()

            result = response.data or {}
            if isinstance(result, list):
                result = result[0] if result else {}

            logger.info(
                "[auto_complete_stale_matches] timeout_hours=%s user_id=%s completed=%s notified=%s",
                safe_hours,
                user_id,
                result.get("completed_count", 0),
                result.get("notified_count", 0),
            )
            return result
        except Exception as e:
            if _is_missing_rpc_error(e, "auto_complete_stale_matches"):
                logger.warning(
                    "[auto_complete_stale_matches] RPC function missing in DB. "
                    "Apply migration 20260315_auto_complete_stale_matches_24h.sql to enable auto-complete."
                )
                return {
                    "timeout_hours": max(int(timeout_hours), 1),
                    "completed_count": 0,
                    "notified_count": 0,
                    "target_user_id": user_id,
                    "migration_required": True,
                }
            logger.exception("[auto_complete_stale_matches] ERROR timeout_hours=%s user_id=%s | %s", timeout_hours, user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

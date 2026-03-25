from fastapi import HTTPException
from fastapi import UploadFile
import base64
import binascii
import logging
import uuid
from API.core.supabase_client import supabase_client, supabase_admin_client, USERS_TABLE
from API.schemas.profile_schemas import ProfileUpdate

logger = logging.getLogger("fairplay.api.profile")
AVATAR_BUCKET = "Avator"
_ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _write_client():
    return supabase_admin_client or supabase_client


def _resolve_public_url(public_url_result) -> str:
    if isinstance(public_url_result, str):
        return public_url_result
    if isinstance(public_url_result, dict):
        return public_url_result.get("publicUrl") or public_url_result.get("public_url") or ""
    return ""


def _upload_avatar_bytes(user_id: int, payload: bytes, content_type: str, filename_hint: str = "avatar") -> str:
    ext = _ALLOWED_IMAGE_TYPES.get(content_type.lower()) if content_type else None
    if not ext:
        raise HTTPException(status_code=400, detail="Unsupported avatar format. Use JPEG, PNG, or WEBP")

    object_path = f"users/{user_id}/{filename_hint}_{uuid.uuid4().hex}.{ext}"
    storage = _write_client().storage.from_(AVATAR_BUCKET)

    try:
        storage.upload(object_path, payload, {"content-type": content_type, "upsert": "true"})
    except TypeError:
        storage.upload(object_path, payload)
    except Exception as exc:
        logger.exception("Avatar upload failed for user_id=%s bucket=%s path=%s", user_id, AVATAR_BUCKET, object_path)
        raise HTTPException(status_code=500, detail=f"Avatar upload failed: {exc}")

    public_url = _resolve_public_url(storage.get_public_url(object_path))
    if not public_url:
        raise HTTPException(status_code=500, detail="Avatar uploaded but public URL could not be generated")
    return public_url


def _upload_avatar_data_uri(user_id: int, avatar_value: str) -> str:
    if not avatar_value.startswith("data:"):
        return avatar_value

    try:
        header, encoded = avatar_value.split(",", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid avatar data URI")

    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Avatar data URI must be base64 encoded")

    content_type = header.replace("data:", "").replace(";base64", "").strip().lower()
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported avatar format. Use JPEG, PNG, or WEBP")

    try:
        avatar_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 avatar payload")

    if len(avatar_bytes) == 0:
        raise HTTPException(status_code=400, detail="Avatar image is empty")
    if len(avatar_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Avatar too large. Max size is 10MB")

    return _upload_avatar_bytes(user_id, avatar_bytes, content_type, filename_hint="avatar_signup")


def _safe_user(user: dict) -> dict:
    if "username" in user:
        user["name"] = user.pop("username")
    return user

class ProfileService:
    @staticmethod
    async def save_avatar_from_signup(user_id: int, avatar_value: str):
        if not avatar_value:
            return None
        avatar_url = _upload_avatar_data_uri(user_id, avatar_value)
        update_resp = _write_client().table(USERS_TABLE).update({"avatar": avatar_url}).eq("id", user_id).execute()
        if not update_resp.data:
            raise HTTPException(status_code=500, detail="Failed to save avatar URL")
        return _safe_user(update_resp.data[0])

    @staticmethod
    async def upload_avatar_file(user_id: int, avatar_file: UploadFile):
        if not avatar_file:
            raise HTTPException(status_code=400, detail="No avatar file uploaded")

        content_type = (avatar_file.content_type or "").lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported avatar format. Use JPEG, PNG, or WEBP")

        payload = await avatar_file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Avatar file is empty")
        if len(payload) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Avatar too large. Max size is 10MB")

        avatar_url = _upload_avatar_bytes(user_id, payload, content_type, filename_hint="avatar_upload")
        update_resp = _write_client().table(USERS_TABLE).update({"avatar": avatar_url}).eq("id", user_id).execute()
        if not update_resp.data:
            raise HTTPException(status_code=500, detail="Failed to update avatar")
        return _safe_user(update_resp.data[0])

    @staticmethod
    async def get_profile(user_id: int):
        try:
            response = supabase_client.table(USERS_TABLE).select("*").eq("id", user_id).execute()
            if response.data:
                return _safe_user(response.data[0])
            raise HTTPException(status_code=404, detail="User not found")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_profile(user_id: int, data: ProfileUpdate):
        try:
            check = supabase_client.table(USERS_TABLE).select("*").eq("id", user_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="User not found")

            current_user = check.data[0]
            update_data = data.model_dump(exclude_unset=True)

            fname = update_data.pop("fname", None)
            lname = update_data.pop("lname", None)

            if fname is not None or lname is not None:
                existing_name = (current_user.get("username") or current_user.get("name") or "").strip()
                existing_parts = existing_name.split(" ", 1) if existing_name else ["", ""]
                existing_first = existing_parts[0] if len(existing_parts) > 0 else ""
                existing_last = existing_parts[1] if len(existing_parts) > 1 else ""

                final_first = (fname if fname is not None else existing_first).strip()
                final_last = (lname if lname is not None else existing_last).strip()
                combined_name = f"{final_first} {final_last}".strip()

                if combined_name:
                    update_data["username"] = combined_name

            elif "name" in update_data:
                update_data["username"] = update_data.pop("name")

            if "avatar" in update_data and isinstance(update_data["avatar"], str) and update_data["avatar"].startswith("data:"):
                update_data["avatar"] = _upload_avatar_data_uri(user_id, update_data["avatar"])

            response = supabase_client.table(USERS_TABLE).update(update_data).eq("id", user_id).execute()
            if response.data:
                return _safe_user(response.data[0])
            raise HTTPException(status_code=500, detail="Failed to update profile")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

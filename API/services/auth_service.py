from fastapi import HTTPException, status
import logging
from API.core.security import get_password_hash, verify_password, create_access_token
from API.schemas.auth_schemas import UserCreate, UserLogin, ChangePassword
from API.core.supabase_client import supabase_client, supabase_admin_client, USERS_TABLE
from API.services.profile_service import ProfileService
from datetime import timedelta

logger = logging.getLogger("fairplay.api.auth")

class AuthService:
    @staticmethod
    async def signup(user_data: UserCreate):
        try:
            logger.info(
                "[signup] START email=%s fname=%r lname=%r name=%r",
                user_data.email, user_data.fname, user_data.lname, user_data.name,
            )

            # Build full name: prefer explicit name field, fall back to fname+lname
            if user_data.fname and user_data.lname:
                user_data.name = f"{user_data.fname} {user_data.lname}".strip()
                logger.debug("[signup] name derived from fname+lname: %r", user_data.name)
            elif not user_data.name or user_data.name.lower() in ("undefined undefined", "none none", ""):
                logger.warning("[signup] Rejected — name field is blank/invalid: %r", user_data.name)
                raise HTTPException(status_code=422, detail="name is required")

            logger.debug("[signup] Resolved name=%r for email=%s", user_data.name, user_data.email)

            # Check if user already exists in public.users
            logger.debug("[signup] Checking existing user in public.users for email=%s", user_data.email)
            existing = supabase_client.table(USERS_TABLE).select("id").eq("email", user_data.email).execute()
            if existing.data:
                logger.warning("[signup] Duplicate email rejected: %s", user_data.email)
                raise HTTPException(status_code=400, detail="Email already registered")
            logger.debug("[signup] Email %s not in public.users — proceeding", user_data.email)

            hashed_password = get_password_hash(user_data.password)
            logger.debug("[signup] Password hashed OK for email=%s", user_data.email)

            if supabase_admin_client is not None:
                logger.info("[signup] Admin client available — using admin.create_user path")
                # ── Admin path: create confirmed Auth user, then upsert profile row ──
                try:
                    auth_resp = supabase_admin_client.auth.admin.create_user({
                        "email": user_data.email,
                        "password": user_data.password,
                        "email_confirm": True,
                        "user_metadata": {
                            "name": user_data.name,
                            "username": user_data.name,
                        },
                    })
                    logger.info("[signup] admin.create_user succeeded for email=%s", user_data.email)
                except Exception as auth_exc:
                    err = str(auth_exc).lower()
                    logger.error(
                        "[signup] admin.create_user FAILED for email=%s | type=%s | detail=%s",
                        user_data.email, type(auth_exc).__name__, auth_exc, exc_info=True,
                    )
                    if any(k in err for k in ("already registered", "already exists", "duplicate", "unique")):
                        raise HTTPException(status_code=400, detail="Email already registered")
                    raise HTTPException(status_code=500, detail=f"Auth user creation failed: {auth_exc}")

                auth_user = getattr(auth_resp, "user", None)
                auth_user_id = str(getattr(auth_user, "id", "")) if auth_user else ""
                logger.debug("[signup] Auth user_id from admin response: %r", auth_user_id)
                if not auth_user_id:
                    logger.error("[signup] admin.create_user returned no user ID for email=%s | resp=%r", user_data.email, auth_resp)
                    raise HTTPException(status_code=500, detail="Supabase Auth user created but returned no ID")

                user_dict = {
                    "auth_user_id": auth_user_id,
                    "username": user_data.name,
                    "email": user_data.email,
                    "password_hash": hashed_password,
                    "avatar": None,
                }
                logger.debug("[signup] Upserting public.users row: %s", {k: v for k, v in user_dict.items() if k != "password_hash"})
                response = supabase_admin_client.table(USERS_TABLE).upsert(user_dict, on_conflict="email").execute()
                logger.debug("[signup] Upsert response row count: %d", len(response.data) if response.data else 0)

                if not response.data:
                    logger.error("[signup] Upsert returned empty data for email=%s", user_data.email)
                    raise HTTPException(status_code=500, detail="Failed to persist user profile")

                user = response.data[0]
                logger.info("[signup] Admin path complete — public.users id=%s email=%s", user.get("id"), user_data.email)

            else:
                logger.info("[signup] No admin client — using anon auth.sign_up fallback path")
                # ── Fallback: use anon auth.sign_up → trigger creates public.users row ──
                try:
                    auth_resp = supabase_client.auth.sign_up({
                        "email": user_data.email,
                        "password": user_data.password,
                        "options": {
                            "data": {
                                "name": user_data.name,
                                "username": user_data.name,
                            }
                        },
                    })
                    logger.info("[signup] anon auth.sign_up call completed for email=%s", user_data.email)
                except Exception as auth_exc:
                    err = str(auth_exc).lower()
                    logger.error(
                        "[signup] auth.sign_up FAILED for email=%s | type=%s | detail=%s",
                        user_data.email, type(auth_exc).__name__, auth_exc, exc_info=True,
                    )
                    if any(k in err for k in ("already registered", "already exists", "duplicate", "unique")):
                        raise HTTPException(status_code=400, detail="Email already registered")
                    raise HTTPException(status_code=500, detail=f"Auth signup failed: {auth_exc}")

                auth_user = getattr(auth_resp, "user", None)
                auth_user_id = str(auth_user.id) if auth_user else ""
                logger.debug("[signup] anon sign_up auth_user_id=%r confirmed=%r", auth_user_id, getattr(auth_user, "email_confirmed_at", None))
                if not auth_user_id:
                    logger.error("[signup] anon sign_up returned no user for email=%s | resp=%r", user_data.email, auth_resp)
                    raise HTTPException(status_code=500, detail="Failed to create user via Supabase Auth")

                # The handle_auth_user_created trigger fires synchronously and creates the
                # public.users row. SELECT policy is USING(true) so anon client can read it.
                logger.debug("[signup] Querying public.users by auth_user_id=%s", auth_user_id)
                profile_resp = supabase_client.table(USERS_TABLE).select("*").eq("auth_user_id", auth_user_id).execute()
                logger.debug("[signup] Trigger-created profile rows found: %d", len(profile_resp.data) if profile_resp.data else 0)

                if profile_resp.data:
                    user = profile_resp.data[0]
                    # Ensure password_hash is persisted (trigger does this if migration 00007 is applied)
                    if not user.get("password_hash"):
                        logger.warning("[signup] Trigger row missing password_hash — patching for auth_user_id=%s", auth_user_id)
                        supabase_client.table(USERS_TABLE).update({"password_hash": hashed_password}).eq("auth_user_id", auth_user_id).execute()
                        user["password_hash"] = hashed_password
                    logger.info("[signup] Trigger path complete — public.users id=%s", user.get("id"))
                else:
                    # Trigger not installed yet — insert directly (RLS INSERT policy allows it)
                    logger.warning("[signup] No trigger row found — inserting directly into public.users for auth_user_id=%s", auth_user_id)
                    user_dict = {
                        "auth_user_id": auth_user_id,
                        "username": user_data.name,
                        "email": user_data.email,
                        "password_hash": hashed_password,
                        "avatar": None,
                    }
                    insert_resp = supabase_client.table(USERS_TABLE).insert(user_dict).execute()
                    logger.debug("[signup] Direct insert row count: %d", len(insert_resp.data) if insert_resp.data else 0)
                    if not insert_resp.data:
                        logger.error("[signup] Direct insert returned no data for email=%s", user_data.email)
                        raise HTTPException(status_code=500, detail="Failed to create user profile")
                    user = insert_resp.data[0]

            if "username" in user:
                user["name"] = user.pop("username")

            # Optional avatar data URI on signup: upload to Supabase storage bucket and persist public URL.
            if user_data.avatar:
                logger.info("[signup] Avatar provided, uploading for user_id=%s", user.get("id"))
                user = await ProfileService.save_avatar_from_signup(user["id"], user_data.avatar)

            logger.info("[signup] SUCCESS email=%s public_id=%s name=%r", user_data.email, user.get("id"), user.get("name"))
            return user

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[signup] UNEXPECTED ERROR for email=%s | type=%s | detail=%s", user_data.email, type(e).__name__, e)
            raise HTTPException(status_code=500, detail=f"Signup error: {str(e)}")

    @staticmethod
    async def login(user_data: UserLogin):
        try:
            logger.info("[login] START email=%s", user_data.email)

            response = supabase_client.table(USERS_TABLE).select("*").eq("email", user_data.email).execute()
            logger.debug("[login] public.users lookup rows=%d", len(response.data) if response.data else 0)

            if not response.data:
                logger.warning("[login] No user found for email=%s", user_data.email)
                raise HTTPException(status_code=401, detail="Invalid credentials")

            user = response.data[0]
            logger.debug("[login] Found user id=%s has_password_hash=%s", user.get("id"), bool(user.get("password_hash")))

            if not user.get("password_hash"):
                logger.warning("[login] User id=%s email=%s has no password_hash — Supabase Auth-only account", user.get("id"), user_data.email)
                raise HTTPException(status_code=401, detail="This account is managed by Supabase Auth. Please sign in from the app.")

            if not verify_password(user_data.password, user["password_hash"]):
                logger.warning("[login] Password mismatch for user id=%s email=%s", user.get("id"), user_data.email)
                raise HTTPException(status_code=401, detail="Invalid credentials")

            access_token = create_access_token({"sub": str(user["id"])})
            if "username" in user:
                user["name"] = user.pop("username")
            logger.info("[login] SUCCESS email=%s user_id=%s", user_data.email, user.get("id"))
            return access_token, user
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[login] UNEXPECTED ERROR for email=%s | type=%s | detail=%s", user_data.email, type(e).__name__, e)
            raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

    @staticmethod
    async def change_password(user_id: int, data: ChangePassword):
        try:
            logger.info("[change_password] START user_id=%s", user_id)
            response = supabase_client.table(USERS_TABLE).select("*").eq("id", user_id).execute()

            if not response.data:
                logger.warning("[change_password] User not found id=%s", user_id)
                raise HTTPException(status_code=404, detail="User not found")

            user = response.data[0]
            if not verify_password(data.old_password, user["password_hash"]):
                logger.warning("[change_password] Old password mismatch for user_id=%s", user_id)
                raise HTTPException(status_code=401, detail="Invalid old password")

            new_hash = get_password_hash(data.new_password)
            update_response = supabase_client.table(USERS_TABLE).update({"password_hash": new_hash}).eq("id", user_id).execute()

            if update_response.data:
                logger.info("[change_password] SUCCESS user_id=%s", user_id)
                return update_response.data[0]
            logger.error("[change_password] Update returned empty data for user_id=%s", user_id)
            raise HTTPException(status_code=500, detail="Failed to change password")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[change_password] UNEXPECTED ERROR user_id=%s | %s", user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def change_password_by_email(data: ChangePassword):
        try:
            if not data.email:
                raise HTTPException(status_code=400, detail="email is required when authorization token is missing")

            logger.info("[change_password_by_email] START email=%s", data.email)
            response = supabase_client.table(USERS_TABLE).select("*").eq("email", data.email).execute()

            if not response.data:
                logger.warning("[change_password_by_email] User not found email=%s", data.email)
                raise HTTPException(status_code=404, detail="User not found")

            user = response.data[0]
            if not user.get("password_hash"):
                logger.warning("[change_password_by_email] User has no password_hash email=%s", data.email)
                raise HTTPException(status_code=401, detail="This account cannot change password via this method")

            if not verify_password(data.old_password, user["password_hash"]):
                logger.warning("[change_password_by_email] Old password mismatch email=%s", data.email)
                raise HTTPException(status_code=401, detail="Invalid old password")

            new_hash = get_password_hash(data.new_password)
            update_response = supabase_client.table(USERS_TABLE).update({"password_hash": new_hash}).eq("id", user["id"]).execute()

            if update_response.data:
                logger.info("[change_password_by_email] SUCCESS email=%s user_id=%s", data.email, user["id"])
                return update_response.data[0]
            logger.error("[change_password_by_email] Update returned empty data email=%s", data.email)
            raise HTTPException(status_code=500, detail="Failed to change password")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[change_password_by_email] UNEXPECTED ERROR email=%s | %s", data.email, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def get_user(user_id: int):
        try:
            logger.debug("[get_user] Fetching user_id=%s", user_id)
            response = supabase_client.table(USERS_TABLE).select("*").eq("id", user_id).execute()
            if response.data:
                user = response.data[0]
                if "username" in user:
                    user["name"] = user.pop("username")
                logger.debug("[get_user] Found user_id=%s email=%s", user_id, user.get("email"))
                return user
            logger.warning("[get_user] User not found id=%s", user_id)
            raise HTTPException(status_code=404, detail="User not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[get_user] ERROR user_id=%s | %s", user_id, e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @staticmethod
    async def update_profile(user_id: int, data: dict):
        try:
            # Only allow specific fields to be updated
            update_data = {}
            if "name" in data:
                update_data["username"] = data["name"]
            if "avatar" in data:
                update_data["avatar"] = data["avatar"]
            
            response = supabase_client.table(USERS_TABLE).update(update_data).eq("id", user_id).execute()
            if response.data:
                user = response.data[0]
                user["name"] = user.pop("username")
                return user
            raise HTTPException(status_code=500, detail="Failed to update profile")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


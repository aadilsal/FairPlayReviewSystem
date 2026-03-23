from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
import logging
from starlette.requests import Request
from API.core.security import decode_access_token
from API.core.supabase_client import supabase_client, USERS_TABLE

logger = logging.getLogger("fairplay.api.auth_dep")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def _extract_bearer_token(request: Request, oauth_token: str | None) -> str | None:
    """
    Resolve JWT token from common locations.
    Priority:
    1) OAuth2 Authorization Bearer token
    2) X-Access-Token / access_token headers
    3) access_token cookie
    4) access_token query param
    """
    if oauth_token:
        return oauth_token

    header_candidates = [
        request.headers.get("x-access-token"),
        request.headers.get("access_token"),
        request.headers.get("token"),
    ]
    for candidate in header_candidates:
        if candidate:
            return candidate.strip()

    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token.strip()

    query_token = request.query_params.get("access_token")
    if query_token:
        return query_token.strip()

    return None

async def get_current_user(request: Request, oauth_token: str | None = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = _extract_bearer_token(request, oauth_token)
    logger.debug("Validating token (first 20 chars): %s...", token[:20] if token else "<empty>")

    if not token:
        logger.warning("No token found in Authorization, alternate headers, cookie, or query")
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        logger.warning("Token decode returned None — token invalid or expired")
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        logger.warning("JWT payload missing 'sub' claim | payload=%s", payload)
        raise credentials_exception

    logger.debug("Token valid — looking up user_id=%s", user_id)

    try:
        response = supabase_client.table(USERS_TABLE).select("*").eq("id", int(user_id)).execute()
        if response.data:
            logger.debug("Auth dependency resolved user_id=%s email=%s", user_id, response.data[0].get("email"))
            return response.data[0]
        logger.warning("Token valid but no user row found for user_id=%s", user_id)
        raise credentials_exception
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in get_current_user for user_id=%s | %s", user_id, exc)
        raise credentials_exception


async def get_current_user_optional(request: Request, oauth_token: str | None = Depends(oauth2_scheme)):
    """Best-effort auth resolution. Returns None when no/invalid token is provided."""
    token = _extract_bearer_token(request, oauth_token)
    if not token:
        return None

    payload = decode_access_token(token)
    if payload is None:
        return None

    user_id: str = payload.get("sub")
    if user_id is None:
        return None

    try:
        response = supabase_client.table(USERS_TABLE).select("*").eq("id", int(user_id)).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception:
        return None

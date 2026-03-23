from datetime import datetime, timedelta
from typing import Optional
import logging
import hashlib
import bcrypt
from jose import jwt, JWTError
from passlib.context import CryptContext
from passlib.exc import MissingBackendError
from API.core.config import settings

logger = logging.getLogger("fairplay.api.security")

# Keep Passlib on pbkdf2_sha256 only to avoid passlib<->bcrypt backend warnings.
# Legacy bcrypt hashes are verified via direct bcrypt.checkpw below.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def _normalize_password(password: str) -> str:
    """
    Accept arbitrarily long passwords safely.

    For very long passwords, pre-hash with SHA-256 to a fixed-length representation
    to avoid backend limits (notably bcrypt's 72-byte input limit) while keeping
    deterministic verification.
    """
    if not isinstance(password, str):
        raise ValueError("password must be a string")

    password_bytes = password.encode("utf-8")
    if len(password_bytes) <= 72:
        return password

    digest = hashlib.sha256(password_bytes).hexdigest()
    return f"sha256${digest}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def verify_password(plain_password, hashed_password):
    is_bcrypt_hash = isinstance(hashed_password, str) and hashed_password.startswith(("$2a$", "$2b$", "$2y$"))
    if is_bcrypt_hash:
        plain_candidates: list[str] = []
        if isinstance(plain_password, str):
            plain_candidates.append(plain_password)
            normalized_plain = _normalize_password(plain_password)
            if normalized_plain not in plain_candidates:
                plain_candidates.append(normalized_plain)
            if len(plain_password.encode("utf-8")) > 72:
                truncated = plain_password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
                plain_candidates.append(truncated)

        for candidate in plain_candidates:
            try:
                if bcrypt.checkpw(candidate.encode("utf-8"), hashed_password.encode("utf-8")):
                    return True
            except ValueError:
                continue
        return False

    normalized_password = _normalize_password(plain_password)
    try:
        return pwd_context.verify(normalized_password, hashed_password)
    except ValueError as exc:
        raise exc
    except MissingBackendError as exc:
        logger.error("Password verify backend unavailable: %s", exc)
        raise exc


def get_password_hash(password):
    normalized_password = _normalize_password(password)
    try:
        return pwd_context.hash(normalized_password)
    except MissingBackendError as exc:
        logger.error("Password hash backend unavailable: %s", exc)
        raise exc


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None

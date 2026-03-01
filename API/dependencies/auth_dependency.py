from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from API.core.security import decode_access_token
from API.core.supabase_client import supabase_client, USERS_TABLE

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    try:
        response = supabase_client.table(USERS_TABLE).select("*").eq("id", int(user_id)).execute()
        if response.data:
            return response.data[0]
        raise credentials_exception
    except Exception:
        raise credentials_exception

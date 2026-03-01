from fastapi.responses import JSONResponse
from typing import Any, Optional

def success_response(data: Any = None, message: str = "Operation successful"):
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": data,
        "message": message
    })

def error_response(message: str, status_code: int = 400, data: Optional[Any] = None):
    return JSONResponse(status_code=status_code, content={
        "status": "error",
        "data": data,
        "message": message
    })

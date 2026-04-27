from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import time
import uuid
from API.routes import auth_routes, match_routes, review_routes, notification_routes, profile_routes, detection_routes, health_routes, video_proxy_routes
from API.core.config import settings
from utils.audio_extractor import is_ffmpeg_available

app = FastAPI(
    title="FairPlayReviewSystem API",
    version="1.1.0",
    description=(
        "Cricket review backend with match lifecycle management, "
        "24-hour stale in-progress auto-completion, and heartbeat support for active matches."
    ),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("fairplay.api")


@app.on_event("startup")
async def startup_diagnostics():
    ffmpeg_ok = is_ffmpeg_available(settings.FFMPEG_BINARY or None)
    if ffmpeg_ok:
        logger.info("Startup diagnostics: ffmpeg available for snick detection")
    else:
        logger.warning("Startup diagnostics: ffmpeg not available, snick detection will run in fallback mode")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_response_logger(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    logger.info(
        "[%s] --> %s %s  client=%s  content-type=%s  content-length=%s",
        request_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        request.headers.get("content-type", "-"),
        request.headers.get("content-length", "-"),
    )

    # Log query params if present
    if request.query_params:
        logger.debug("[%s] query_params=%s", request_id, dict(request.query_params))

    # Warn if Authorization header is missing on protected routes
    auth_header = request.headers.get("authorization", "")
    if not auth_header and request.url.path not in ("/api/auth/signup", "/api/auth/login", "/api/health"):
        logger.warning("[%s] No Authorization header on %s %s", request_id, request.method, request.url.path)

    response = await call_next(request)

    process_time_ms = (time.perf_counter() - start_time) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "[%s] <-- %s %s  status=%s  (%.2f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        process_time_ms,
    )

    response.headers["X-Request-ID"] = request_id
    return response

# Routers
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Auth"])
app.include_router(match_routes.router, prefix="/api/matches", tags=["Matches"])
app.include_router(review_routes.router, prefix="/api/reviews", tags=["Reviews"])
app.include_router(notification_routes.router, prefix="/api/notifications", tags=["Notifications"])
app.include_router(profile_routes.router, prefix="/api/profile", tags=["Profile"])
app.include_router(detection_routes.router, prefix="/api", tags=["Detection"])
app.include_router(health_routes.router, prefix="/api/health", tags=["Health"])
app.include_router(video_proxy_routes.router, tags=["Video"])

# Global Exception Handler
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    message = detail if isinstance(detail, str) else "Request failed"

    logger.warning(
        "HTTP ERROR on %s %s | status=%s | detail=%s",
        request.method,
        request.url.path,
        exc.status_code,
        detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "data": None,
            "message": message,
            "detail": detail,
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "VALIDATION ERROR on %s %s | errors=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "data": None,
            "message": "Validation error",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "UNHANDLED EXCEPTION on %s %s | type=%s | detail=%s",
        request.method,
        request.url.path,
        type(exc).__name__,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "data": None,
            "message": str(exc)
        },
    )

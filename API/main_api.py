from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from API.routes import auth_routes, match_routes, review_routes, notification_routes, profile_routes, detection_routes, health_routes

app = FastAPI(title="FairPlayReviewSystem API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Auth"])
app.include_router(match_routes.router, prefix="/api/matches", tags=["Matches"])
app.include_router(review_routes.router, prefix="/api/reviews", tags=["Reviews"])
app.include_router(notification_routes.router, prefix="/api/notifications", tags=["Notifications"])
app.include_router(profile_routes.router, prefix="/api/profile", tags=["Profile"])
app.include_router(detection_routes.router, prefix="/api", tags=["Detection"])
app.include_router(health_routes.router, prefix="/api/health", tags=["Health"])

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "data": None,
            "message": str(exc)
        },
    )

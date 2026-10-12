"""
FallGuard AI - Central FastAPI Application Entrypoint
AI-Based Wearable Fall Detection & Real-Time Activity Monitoring System
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.core.config import settings
from backend.core.logger import app_logger
from backend.database.init_db import init_db
from backend.api.routes import auth, predictions, falls, activities, dashboard, simulation, model, ws, contacts

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and graceful shutdown."""
    app_logger.info("=" * 60)
    app_logger.info(f"[START] Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    app_logger.info(f"[CONFIG] Environment: {settings.ENVIRONMENT} | Debug: {settings.DEBUG}")
    app_logger.info("=" * 60)
    
    # Validate production JWT secret
    if settings.ENVIRONMENT == "production":
        _ = settings.resolved_secret

    # Initialize DB & seed default accounts
    init_db()
    
    yield
    
    app_logger.info(f"[STOP] Shutting down {settings.APP_NAME}...")

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Based Wearable Fall Detection and Real-Time Activity Monitoring System API",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(auth.router, prefix="/api")
app.include_router(contacts.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(falls.router, prefix="/api")
app.include_router(activities.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(ws.router)

# Health Check
@app.get("/api/health", tags=["System"])
def health_check():
    """Health check endpoint for container / load balancer probes."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

# Mount Frontend Static Directory
STATIC_DIR = settings.STATIC_DIR
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"message": "FallGuard AI Backend Running. Dashboard index.html not found."})

    @app.get("/login", include_in_schema=False)
    @app.get("/login.html", include_in_schema=False)
    async def serve_login():
        login_path = STATIC_DIR / "login.html"
        if login_path.exists():
            return FileResponse(login_path)
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/signup", include_in_schema=False)
    @app.get("/signup.html", include_in_schema=False)
    async def serve_signup():
        signup_path = STATIC_DIR / "signup.html"
        if signup_path.exists():
            return FileResponse(signup_path)
        return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )

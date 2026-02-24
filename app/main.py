"""
FastAPI application entry point.

Configures middleware, lifespan events, and mounts all routers.
Run locally: uvicorn app.main:app --reload
Production:  gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.routes import approvals, health, runs
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.security import limiter

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    setup_logging()
    logger.info(
        "app_starting",
        environment=settings.app_env,
        database=settings.database_url[:30] + "...",
    )

    # Create output directories
    Path("./output/images").mkdir(parents=True, exist_ok=True)

    yield

    logger.info("app_shutting_down")


app = FastAPI(
    title="AI News Summarizer",
    description="LangGraph multi-agent pipeline for AI/ML news curation and publishing",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url="/redoc" if settings.app_env != "production" else None,
)

# ── Middleware ──────────────────────────────────────────────
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ──────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Static files (for serving generated images locally) ─────
Path("output/images").mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory="output/images"), name="images")

# ── Routes ─────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(runs.router, prefix="/api/v1")
app.include_router(approvals.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "AI News Summarizer",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/healthz/",
    }

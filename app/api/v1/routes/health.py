"""Health check endpoints — used by Railway's healthcheck and monitoring."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        environment=settings.app_env,
        database="connected",  # TODO: add actual DB ping
    )

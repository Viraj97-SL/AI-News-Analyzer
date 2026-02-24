"""
Pydantic v2 schemas for API request/response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Pipeline trigger ────────────────────────────────────────
class TriggerRequest(BaseModel):
    trigger_type: Literal["manual"] = "manual"


class TriggerResponse(BaseModel):
    run_id: str
    status: str = "started"
    message: str = "Pipeline execution started in background"


# ── Run status ──────────────────────────────────────────────
class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    trigger_type: str
    started_at: datetime
    completed_at: datetime | None = None
    total_tokens: int = 0
    total_cost: float = 0.0
    current_step: str | None = None
    error_log: str | None = None


# ── Approval ────────────────────────────────────────────────
class ApprovalRequest(BaseModel):
    action: Literal["approve", "reject"]
    feedback: str = Field(default="", max_length=1000)


class ApprovalResponse(BaseModel):
    run_id: str
    action: str
    status: str
    message: str


# ── Health check ────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    environment: str
    database: str = "connected"

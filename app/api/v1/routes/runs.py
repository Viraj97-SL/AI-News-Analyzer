"""
Pipeline trigger and status endpoints.

POST /api/v1/runs/trigger — kick off a pipeline run (background task)
GET  /api/v1/runs/{run_id} — poll run status
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agents.graph import build_graph
from app.agents.state import PipelineState
from app.api.v1.deps import AppSettings, AuthenticatedUser
from app.core.logging import get_logger
from app.schemas.schemas import RunStatusResponse, TriggerRequest, TriggerResponse

router = APIRouter(prefix="/runs", tags=["runs"])
logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

# In-memory run status tracker (replace with DB queries in production)
_run_status: dict[str, dict] = {}


async def execute_pipeline(run_id: str, trigger_type: str = "manual") -> None:
    """Background task: execute the full pipeline graph."""
    try:
        _run_status[run_id] = {"status": "running", "current_step": "starting"}
        graph = build_graph()

        initial_state: PipelineState = {
            "run_id": run_id,
            "trigger_type": trigger_type,
            "raw_articles": [],
            "deduplicated_articles": [],
            "summaries": [],
            "newsletter_html": "",
            "linkedin_draft": "",
            "image_paths": [],
            "approval_status": "pending",
            "feedback": "",
            "error_log": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "current_step": "starting",
        }

        config = {"configurable": {"thread_id": run_id}}
        result = await graph.ainvoke(initial_state, config)

        _run_status[run_id] = {
            "status": result.get("approval_status", "completed"),
            "current_step": result.get("current_step", "finished"),
        }
        logger.info("pipeline_completed", run_id=run_id)

    except Exception as e:
        logger.error("pipeline_failed", run_id=run_id, error=str(e))
        _run_status[run_id] = {"status": "failed", "error": str(e)}


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_run(
    request: Request,
    background_tasks: BackgroundTasks,
    _api_key: AuthenticatedUser,
) -> TriggerResponse:
    """Trigger a new pipeline run. Returns immediately with a run_id for polling."""
    run_id = str(uuid.uuid4())
    background_tasks.add_task(execute_pipeline, run_id, "manual")
    logger.info("pipeline_triggered", run_id=run_id, trigger="manual")

    return TriggerResponse(run_id=run_id, status="started")


@router.get("/{run_id}", response_model=dict)
async def get_run_status(run_id: str, _api_key: AuthenticatedUser) -> dict:
    """Get the current status of a pipeline run."""
    if run_id not in _run_status:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"run_id": run_id, **_run_status[run_id]}

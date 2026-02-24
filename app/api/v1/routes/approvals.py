"""
Human-in-the-loop approval endpoints.

POST /api/v1/approvals/{run_id} — approve/reject a pending run
GET  /api/v1/approvals/via-token   — one-click approve/reject from email link
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.deps import AppSettings, AuthenticatedUser
from app.core.logging import get_logger
from app.core.security import verify_approval_token
from app.schemas.schemas import ApprovalRequest, ApprovalResponse

router = APIRouter(prefix="/approvals", tags=["approvals"])
logger = get_logger(__name__)


@router.post("/{run_id}", response_model=ApprovalResponse)
async def approve_or_reject(
    run_id: str,
    body: ApprovalRequest,
    _api_key: AuthenticatedUser,
) -> ApprovalResponse:
    """
    Approve or reject a pipeline run awaiting human review.

    This resumes the LangGraph from its interrupt() point.
    """
    try:
        from langgraph.types import Command

        from app.agents.graph import build_graph

        graph = build_graph()
        config = {"configurable": {"thread_id": run_id}}

        # Resume the interrupted graph
        result = await graph.ainvoke(
            Command(resume={"action": body.action, "feedback": body.feedback}),
            config,
        )

        logger.info(
            "approval_processed",
            run_id=run_id,
            action=body.action,
            has_feedback=bool(body.feedback),
        )

        return ApprovalResponse(
            run_id=run_id,
            action=body.action,
            status="resumed",
            message=f"Pipeline {body.action}d and resumed.",
        )

    except Exception as e:
        logger.error("approval_error", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/via-token")
async def approve_via_email_token(
    token: str = Query(..., description="HMAC-signed JWT from approval email"),
    settings: AppSettings = None,
) -> ApprovalResponse:
    """
    One-click approval from email link (GET request with signed token).

    The token encodes run_id, action, and expiry.
    """
    payload = verify_approval_token(token, settings)
    run_id = payload["run_id"]
    action = payload["action"]

    # TODO: Enforce one-time-use by storing token JTI in DB and checking here

    logger.info("email_approval_received", run_id=run_id, action=action)

    # Delegate to the same approval logic
    request = ApprovalRequest(action=action)
    return await approve_or_reject(run_id, request, _api_key="email-token")

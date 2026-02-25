"""
Human approval node — pauses the pipeline for review before publishing.

Uses LangGraph's interrupt() to persist state and wait for a human decision.
The approval email contains signed approve/reject URLs that resume the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.types import Command, interrupt

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)


def human_approval_node(state: PipelineState) -> Command[Literal["publish", "revise"]]:
    """
    Pause execution and wait for human approval.

    The interrupt() call serialises the preview payload and suspends the graph.
    A FastAPI endpoint (see api/v1/routes/approvals.py) resumes it with:
      graph.invoke(Command(resume={"action": "approve"}), config)
    """
    run_id = state.get("run_id", "unknown")
    logger.info(
        "awaiting_approval",
        run_id=run_id,
        linkedin_chars=len(state.get("linkedin_draft", "")),
        image_count=len(state.get("image_paths", [])),
    )

    # Send approval email with signed approve/reject links before suspending
    try:
        from app.core.config import get_settings
        from app.core.security import create_approval_token
        from app.services.email_service import EmailService

        _settings = get_settings()
        approve_token = create_approval_token(run_id, "approve", _settings)
        reject_token = create_approval_token(run_id, "reject", _settings)
        base = _settings.app_base_url.rstrip("/")
        approve_url = f"{base}/api/v1/approvals/via-token?token={approve_token}"
        reject_url = f"{base}/api/v1/approvals/via-token?token={reject_token}"

        EmailService().send_approval_email(
            run_id=run_id,
            linkedin_preview=state.get("linkedin_draft", ""),
            approve_url=approve_url,
            reject_url=reject_url,
            image_paths=state.get("image_paths", []),  # PASS THE RAW FILE PATHS
        )
        logger.info("approval_email_sent", run_id=run_id)
    except Exception as e:
        logger.error("approval_email_failed", error=str(e))

    # This suspends the graph and returns the payload to the caller
    decision = interrupt(
        {
            "linkedin_draft": state.get("linkedin_draft", ""),
            "newsletter_preview": state.get("newsletter_html", "")[:500],
            "image_count": len(state.get("image_paths", [])),
            "summary_count": len(state.get("summaries", [])),
            "message": "Please review the content and approve or reject with feedback.",
        }
    )

    action = decision.get("action", "reject")
    feedback = decision.get("feedback", "")

    logger.info("approval_decision", action=action, has_feedback=bool(feedback))

    if action == "approve":
        return Command(
            update={"approval_status": "approved", "current_step": "approved"},
            goto="publish",
        )

    return Command(
        update={
            "approval_status": "rejected",
            "feedback": feedback,
            "current_step": "revision_requested",
        },
        goto="revise",
    )

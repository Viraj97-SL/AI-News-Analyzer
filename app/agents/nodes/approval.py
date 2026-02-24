"""
Human approval node — pauses the pipeline for review before publishing.

Uses LangGraph's interrupt() to persist state and wait for a human decision.
The approval email contains signed approve/reject URLs that resume the graph.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command, interrupt

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
    logger.info(
        "awaiting_approval",
        run_id=state.get("run_id"),
        linkedin_chars=len(state.get("linkedin_draft", "")),
        image_count=len(state.get("image_paths", [])),
    )

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

"""
Main supervisor graph — orchestrates the full news pipeline.

Architecture: Manual supervisor pattern using StateGraph + Command routing.
See design doc Section 1 for the full flow diagram.

Flow:
  START → scrape (fan-out) → merge → deduplicate → credibility
  → analyse → summarise → linkedin_gen → image_gen
  → human_approval [interrupt] → (approved) email + linkedin_publish
                                 (rejected) revision loop
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from app.agents.nodes.approval import human_approval_node
from app.agents.nodes.credibility import credibility_node
from app.agents.nodes.image_gen import image_gen_node
from app.agents.nodes.linkedin_gen import linkedin_gen_node
from app.agents.nodes.scraper import (
    merge_results_node,
    scrape_arxiv_node,
    scrape_rss_node,
    scrape_serper_node,
    scrape_tavily_node,
)
from app.agents.nodes.summarizer import (
    analyze_node,
    deduplicate_node,
    summarize_node,
)
from app.agents.state import PipelineState
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Retry policy for scraper nodes (transient failure handling) ──
_scraper_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=2.0,
    backoff_factor=2,
    jitter=True,
)


def _fan_out_scrapers(state: PipelineState) -> list[Send]:
    """Fan out to all scraper nodes in parallel via Send."""
    return [
        Send("scrape_tavily", state),
        Send("scrape_rss", state),
        Send("scrape_arxiv", state),
        Send("scrape_serper", state),
    ]


def _route_after_approval(state: PipelineState) -> Literal["publish", "revise"]:
    """Conditional edge: route based on human approval decision."""
    if state.get("approval_status") == "approved":
        return "publish"
    return "revise"


def build_graph(checkpointer=None) -> StateGraph:
    """
    Construct and compile the full pipeline graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Use InMemorySaver for dev, AsyncPostgresSaver for prod.

    Returns:
        Compiled StateGraph ready for .invoke() or .ainvoke().
    """
    workflow = StateGraph(PipelineState)

    # ── Scraper nodes (parallel fan-out) ────────────────────
    workflow.add_node("scrape_tavily", scrape_tavily_node, retry=_scraper_retry)
    workflow.add_node("scrape_rss", scrape_rss_node, retry=_scraper_retry)
    workflow.add_node("scrape_arxiv", scrape_arxiv_node, retry=_scraper_retry)
    workflow.add_node("scrape_serper", scrape_serper_node, retry=_scraper_retry)
    workflow.add_node("merge_results", merge_results_node)

    # ── Processing nodes (sequential) ───────────────────────
    workflow.add_node("deduplicate", deduplicate_node)
    workflow.add_node("credibility", credibility_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("summarize", summarize_node)

    # ── Content generation ──────────────────────────────────
    workflow.add_node("linkedin_gen", linkedin_gen_node)
    workflow.add_node("image_gen", image_gen_node)

    # ── Human-in-the-loop ───────────────────────────────────
    workflow.add_node("human_approval", human_approval_node)

    # ── Publishing (placeholder nodes — implemented in services) ──
    workflow.add_node("publish", _publish_node)
    workflow.add_node("revise", _revise_node)

    # ── Edges ───────────────────────────────────────────────
    # Fan-out from START to all scrapers
    workflow.add_conditional_edges(START, _fan_out_scrapers)

    # All scrapers → merge
    for scraper in ["scrape_tavily", "scrape_rss", "scrape_arxiv", "scrape_serper"]:
        workflow.add_edge(scraper, "merge_results")

    # Sequential pipeline
    workflow.add_edge("merge_results", "deduplicate")
    workflow.add_edge("deduplicate", "credibility")
    workflow.add_edge("credibility", "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", "linkedin_gen")
    workflow.add_edge("linkedin_gen", "image_gen")
    workflow.add_edge("image_gen", "human_approval")

    # Conditional after approval
    workflow.add_conditional_edges("human_approval", _route_after_approval)
    workflow.add_edge("publish", END)
    workflow.add_edge("revise", "summarize")  # revision loop

    # ── Compile ─────────────────────────────────────────────
    if checkpointer is None:
        checkpointer = InMemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("pipeline_graph_compiled", node_count=len(workflow.nodes))
    return app


# ── Publish / Revise placeholder nodes ──────────────────────
def _publish_node(state: PipelineState) -> dict:
    """Send email newsletter and publish LinkedIn post."""
    # TODO: Wire up email_service.send_newsletter() and linkedin_service.publish_post()
    logger.info(
        "publishing",
        run_id=state["run_id"],
        linkedin_chars=len(state.get("linkedin_draft", "")),
    )
    return {"current_step": "published"}


def _revise_node(state: PipelineState) -> dict:
    """Incorporate human feedback and loop back to summarizer."""
    logger.info(
        "revision_requested",
        run_id=state["run_id"],
        feedback=state.get("feedback", ""),
    )
    return {"current_step": "revising"}

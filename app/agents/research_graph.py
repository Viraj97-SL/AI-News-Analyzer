from __future__ import annotations
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.approval import human_approval_node
from app.agents.nodes.scraper import scrape_arxiv_node
from app.agents.state import PipelineState
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── 1. Placeholder Intelligence Nodes (To be built in Phase 2) ──
def select_paper_node(state: PipelineState) -> dict:
    """Filters ArXiv results and selects the single best paper."""
    logger.info("research_node_running", step="selecting_best_paper")
    return {"current_step": "paper_selected"}


def deep_analysis_node(state: PipelineState) -> dict:
    """Uses Gemini Pro to extract thematic analysis and methodology."""
    logger.info("research_node_running", step="deep_analysis")
    # For now, we populate the draft variables so the approval node doesn't break
    return {
        "linkedin_draft": "🚨 Deep Tech Research Analysis: [Placeholder]",
        "newsletter_html": "<h2>Deep Dive Methodology</h2><p>[Placeholder]</p>",
        "current_step": "analysis_complete"
    }


def paperbanana_visual_node(state: PipelineState) -> dict:
    """Uses PaperBanana Agentic workflow to generate system architecture."""
    logger.info("research_node_running", step="generating_paperbanana_visual")
    return {"image_paths": [], "current_step": "visuals_generated"}


# ── 2. Publishing Nodes ──
def _publish_research_node(state: PipelineState) -> dict:
    logger.info("research_published", run_id=state["run_id"])
    return {"current_step": "published"}


def _revise_research_node(state: PipelineState) -> dict:
    return {"current_step": "revising"}


def _route_after_approval(state: PipelineState) -> Literal["publish", "revise"]:
    if state.get("approval_status") == "approved":
        return "publish"
    return "revise"


# ── 3. Build the Graph ──
def build_research_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(PipelineState)

    # Scrape strictly ArXiv
    workflow.add_node("scrape_arxiv", scrape_arxiv_node)

    # Intelligence Pipeline
    workflow.add_node("select_paper", select_paper_node)
    workflow.add_node("deep_analysis", deep_analysis_node)
    workflow.add_node("paperbanana_visual", paperbanana_visual_node)

    # HITL & Publish
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("publish", _publish_research_node)
    workflow.add_node("revise", _revise_research_node)

    # Edges
    workflow.add_edge(START, "scrape_arxiv")
    workflow.add_edge("scrape_arxiv", "select_paper")
    workflow.add_edge("select_paper", "deep_analysis")
    workflow.add_edge("deep_analysis", "paperbanana_visual")
    workflow.add_edge("paperbanana_visual", "human_approval")

    workflow.add_conditional_edges("human_approval", _route_after_approval)
    workflow.add_edge("publish", END)
    workflow.add_edge("revise", "deep_analysis")  # Re-analyze on rejection

    if checkpointer is None:
        checkpointer = InMemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("research_graph_compiled", node_count=len(workflow.nodes))
    return app
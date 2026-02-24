"""
LangGraph pipeline state — the single source of truth flowing through every node.

Design principle: store raw data in state, format prompts on demand in each node.
Annotated reducers allow parallel scrapers to append concurrently without races.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class NewsArticle(TypedDict):
    title: str
    url: str
    source: str  # e.g. "tavily", "rss:techcrunch", "arxiv", "serper"
    content: str  # full text or abstract
    published_at: str  # ISO-8601
    credibility_score: float  # 0.0-1.0, populated by credibility node
    category: NotRequired[str]  # populated by analyze_node
    relevance_score: NotRequired[float]  # 0.0-1.0, populated by analyze_node


class Summary(TypedDict):
    headline: str
    body: str  # 2-3 paragraph summary
    category: str  # e.g. "LLM", "Computer Vision", "Robotics", "Policy"
    source_urls: list[str]
    credibility_score: float


class PipelineState(TypedDict):
    """Top-level state for the supervisor graph."""

    # ── Run metadata ────────────────────────────────────────
    run_id: str
    trigger_type: Literal["scheduled", "manual"]

    # ── Data pipeline ───────────────────────────────────────
    raw_articles: Annotated[list[NewsArticle], operator.add]
    deduplicated_articles: list[NewsArticle]
    summaries: list[Summary]

    # ── Content generation ──────────────────────────────────
    newsletter_html: str
    linkedin_draft: str
    image_paths: list[str]

    # ── Human-in-the-loop ───────────────────────────────────
    approval_status: Literal["pending", "approved", "rejected"]
    feedback: str  # human feedback on rejection

    # ── Observability ───────────────────────────────────────
    error_log: Annotated[list[str], operator.add]
    total_tokens: int
    total_cost: float
    current_step: str

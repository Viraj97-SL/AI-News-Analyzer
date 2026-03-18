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
    image_paths: list[str]                        # 1200×627 cards used in email newsletter
    carousel_slide_paths: NotRequired[list[str]]  # 1080×1080 PNG slides — sent in approval email preview
    carousel_pdf_path: NotRequired[str]           # Combined PDF — uploaded to LinkedIn as document post

    # ── Human-in-the-loop ───────────────────────────────────
    approval_status: Literal["pending", "approved", "rejected"]
    feedback: str  # human feedback on rejection

    # ── Observability ───────────────────────────────────────
    error_log: Annotated[list[str], operator.add]
    total_tokens: int
    total_cost: float
    current_step: str

    # ── Research Analyst Variables ──────────────────────────
    chosen_research_paper: NotRequired[dict]
    deep_analysis: NotRequired[dict]

    # ── Research Analyst Enhancements ───────────────────────
    research_scores: NotRequired[dict]                     # F8: novelty/clarity/benchmarks/reproducibility 1-10
    hook_score: NotRequired[dict]                          # F1: hook quality scores
    hook_attempts: NotRequired[int]                        # F1: regeneration counter
    benchmark_metrics: NotRequired[list[dict]]             # F5: extracted benchmark data
    benchmark_chart_path: NotRequired[str]                 # F5: matplotlib bar chart PNG path
    architecture_diagram_path: NotRequired[str]            # F6: cropped PDF figure path or ""
    architecture_diagram_b64: NotRequired[str]             # F6: base64-encoded diagram
    architecture_fallback_text: NotRequired[str]           # F6: ASCII box diagram HTML
    prior_art_comparison: NotRequired[dict]                # F7: vs-prior-SOTA structured comparison
    comparison_card_path: NotRequired[str]                 # F7: prior art card PNG path
    research_carousel_pdf_path: NotRequired[str]           # F2: 5-slide carousel PDF
    research_carousel_slide_paths: NotRequired[list[str]]  # F2: individual 1080x1080 PNGs

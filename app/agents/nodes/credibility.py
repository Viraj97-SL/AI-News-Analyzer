"""
Credibility scoring node — three-layer verification system.


Layer 1: Source reputation lookup (domain → score database)

Layer 2: Cross-reference verification (multi-source confirmation)


Layer 3: LLM-based factual consistency (RAG-augmented fact checking)

Final score = 0.4 * source_reputation + 0.3 * cross_reference + 0.3 * factual_consistency
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import NewsArticle, PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Layer 1: Source reputation database ─────────────────────
# Scores 0.0-1.0, sourced from NewsGuard / MBFC / domain authority
# Expand this as you onboard more sources.
SOURCE_REPUTATION: dict[str, float] = {
    # Tier 1 — established tech journalism
    "techcrunch.com": 0.85,
    "venturebeat.com": 0.80,
    "theverge.com": 0.82,
    "wired.com": 0.85,
    "arstechnica.com": 0.88,
    "technologyreview.com": 0.90,
    # Tier 2 — major news
    "reuters.com": 0.95,
    "bbc.com": 0.92,
    "nytimes.com": 0.90,
    "washingtonpost.com": 0.88,
    # Tier 3 — tech blogs & aggregators
    "thenewstack.io": 0.72,
    "medium.com": 0.50,
    "towardsdatascience.com": 0.55,
    "dev.to": 0.45,
    # Tier 4 — research
    "arxiv.org": 0.80,
    "openreview.net": 0.82,
    "nature.com": 0.95,
    "science.org": 0.95,
    # Default for unknown
    "_default": 0.40,
}


def _get_source_reputation(url: str) -> float:
    """Look up domain reputation score."""
    from urllib.parse import urlparse

    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return SOURCE_REPUTATION["_default"]

    # Check exact match, then parent domain
    if domain in SOURCE_REPUTATION:
        return SOURCE_REPUTATION[domain]

    # Try parent domain (e.g., blog.google → google)
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in SOURCE_REPUTATION:
            return SOURCE_REPUTATION[parent]

    return SOURCE_REPUTATION["_default"]


def credibility_node(state: PipelineState) -> dict:
    """
    Score each article's credibility using the three-layer system.

    For MVP, implements Layer 1 (source reputation) fully.
    Layers 2 and 3 are stubbed — fill in with cross-reference and
    LLM consistency checks when ready.
    """
    articles = state.get("deduplicated_articles", [])
    if not articles:
        return {"error_log": ["Credibility: no articles to score"]}

    scored: list[NewsArticle] = []
    for article in articles:
        # Layer 1: source reputation
        source_score = _get_source_reputation(article["url"])

        # Layer 2: cross-reference (stub — returns neutral 0.5)
        # TODO: Use Gemini with Google Search grounding to find corroborating sources
        cross_ref_score = 0.5

        # Layer 3: LLM factual consistency (stub — returns neutral 0.5)
        # TODO: Extract claims → retrieve evidence → assess with Gemini Pro
        factual_score = 0.5

        # Weighted composite
        final_score = (
            0.4 * source_score + 0.3 * cross_ref_score + 0.3 * factual_score
        )

        scored_article = {**article, "credibility_score": round(final_score, 3)}
        scored.append(scored_article)

    above_threshold = sum(1 for a in scored if a["credibility_score"] >= 0.4)
    logger.info(
        "credibility_scored",
        total=len(scored),
        above_threshold=above_threshold,
    )

    return {"deduplicated_articles": scored, "current_step": "credibility_scored"}

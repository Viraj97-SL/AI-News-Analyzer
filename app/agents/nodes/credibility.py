



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
    "zdnet.com": 0.78,
    "cnet.com": 0.76,
    "engadget.com": 0.75,
    "ieee.org": 0.92,
    # Tier 2 — major news
    "reuters.com": 0.95,
    "bbc.com": 0.92,
    "nytimes.com": 0.90,
    "washingtonpost.com": 0.88,
    "ft.com": 0.90,
    "bloomberg.com": 0.90,
    "wsj.com": 0.88,
    "economist.com": 0.90,
    # Tier 3 — AI-specific outlets
    "ai.googleblog.com": 0.82,
    "blog.google": 0.80,
    "openai.com": 0.78,
    "deepmind.google": 0.83,
    "deepmind.com": 0.83,
    "anthropic.com": 0.80,
    "huggingface.co": 0.75,
    "mistral.ai": 0.74,
    "ai-business.org": 0.70,
    # Tier 4 — tech blogs & aggregators
    "thenewstack.io": 0.72,
    "infoq.com": 0.72,
    "medium.com": 0.50,
    "towardsdatascience.com": 0.55,
    "dev.to": 0.45,
    "substack.com": 0.48,
    # Tier 5 — research
    "arxiv.org": 0.80,
    "openreview.net": 0.82,
    "nature.com": 0.95,
    "science.org": 0.95,
    "cell.com": 0.93,
    "acm.org": 0.88,
    "semanticscholar.org": 0.80,
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


_CROSS_REF_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "to", "in", "on", "at", "by", "for", "with", "from", "of", "and", "or", "but",
    "not", "new", "ai", "model", "llm", "using", "how", "why", "what", "says",
    "report", "week", "latest", "update", "2024", "2025", "first", "more", "its",
})


def _cross_reference_score(article: dict, all_articles: list[dict]) -> float:
    """
    Layer 2: Estimate corroboration by counting other articles with significant
    title-keyword overlap from *different* domains.

    Logic:
      - Extract meaningful keywords from the article title
      - Count other articles (from different domains) with 2+ keyword matches
      - More corroboration → higher score (capped at 0.90)
    """
    from urllib.parse import urlparse

    def _domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return url

    title_words = set(article["title"].lower().split()) - _CROSS_REF_STOP_WORDS
    # Keep only words ≥ 4 chars to filter noise
    title_words = {w.strip(".,;:!?\"'") for w in title_words if len(w) >= 4}

    if not title_words:
        return 0.50

    own_domain = _domain(article["url"])
    corroborating_domains: set[str] = set()

    for other in all_articles:
        if other["url"] == article["url"]:
            continue
        other_domain = _domain(other["url"])
        if other_domain == own_domain:
            continue  # same source doesn't count as corroboration

        other_words = set(other["title"].lower().split()) - _CROSS_REF_STOP_WORDS
        other_words = {w.strip(".,;:!?\"'") for w in other_words if len(w) >= 4}
        shared = title_words & other_words

        if len(shared) >= 2:
            corroborating_domains.add(other_domain)

    n = len(corroborating_domains)
    if n == 0:
        return 0.35   # unique story — lower confidence
    if n == 1:
        return 0.55   # one corroboration
    if n == 2:
        return 0.70   # two sources
    if n == 3:
        return 0.80   # three sources
    return min(0.90, 0.80 + (n - 3) * 0.05)  # 4+ sources


def credibility_node(state: PipelineState) -> dict:
    """
    Score each article's credibility using the three-layer system.

    Layer 1: Source domain reputation (expanded database).
    Layer 2: Cross-reference — counts corroborating articles from distinct domains.
    Layer 3: LLM factual consistency (stub — neutral 0.5 until implemented).
    """
    articles = state.get("deduplicated_articles", [])
    if not articles:
        return {"error_log": ["Credibility: no articles to score"]}

    scored: list[NewsArticle] = []
    for article in articles:
        # Layer 1: source reputation
        source_score = _get_source_reputation(article["url"])

        # Layer 2: cross-reference corroboration (active)
        cross_ref_score = _cross_reference_score(article, articles)

        # Layer 3: LLM factual consistency (stub — returns neutral 0.5)
        # TODO: Extract claims → retrieve evidence → assess with Gemini Pro
        factual_score = 0.5

        # Weighted composite
        final_score = 0.4 * source_score + 0.3 * cross_ref_score + 0.3 * factual_score

        scored_article = {**article, "credibility_score": round(final_score, 3)}
        scored.append(scored_article)

    above_threshold = sum(1 for a in scored if a["credibility_score"] >= 0.5)
    logger.info(
        "credibility_scored",
        total=len(scored),
        above_threshold=above_threshold,
        avg_score=round(sum(a["credibility_score"] for a in scored) / len(scored), 3),
    )

    return {"deduplicated_articles": scored, "current_step": "credibility_scored"}

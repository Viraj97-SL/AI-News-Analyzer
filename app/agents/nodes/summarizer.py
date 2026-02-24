"""
Summarizer nodes — deduplication, analysis, and LLM summarization.

Uses tiered model routing:
  - Flash-Lite for topic classification
  - Flash for summarization
  - Pro reserved for complex analysis (called from credibility node)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

if TYPE_CHECKING:
    from app.agents.state import NewsArticle, PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import hash_content

logger = get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════
# Deduplication — content-hash based + title similarity
# ═══════════════════════════════════════════════════════════════
def deduplicate_node(state: PipelineState) -> dict:
    """Remove duplicate articles using content hashing and title overlap."""
    raw = state.get("raw_articles", [])
    seen_hashes: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[NewsArticle] = []

    for article in raw:
        content_hash = hash_content(article["content"])
        title_lower = article["title"].lower().strip()

        # Skip exact content duplicates
        if content_hash in seen_hashes:
            continue

        # Skip near-identical titles (simple approach; upgrade to fuzzy matching later)
        if title_lower in seen_titles:
            continue

        seen_hashes.add(content_hash)
        seen_titles.add(title_lower)
        unique.append(article)

    logger.info(
        "deduplication_complete",
        raw_count=len(raw),
        unique_count=len(unique),
        removed=len(raw) - len(unique),
    )
    return {"deduplicated_articles": unique, "current_step": "deduplicated"}


# ═══════════════════════════════════════════════════════════════
# Analysis — categorise and rank articles by relevance
# ═══════════════════════════════════════════════════════════════
def analyze_node(state: PipelineState) -> dict:
    """
    Categorise articles into topics and rank by significance.
    Uses Gemini Flash-Lite (cheapest tier) for classification.
    """
    articles = state.get("deduplicated_articles", [])
    if not articles:
        return {"error_log": ["Analyze: no articles to process"]}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model_classifier,
            temperature=0,
            google_api_key=settings.google_api_key,
        )

        # Batch classification prompt
        article_list = "\n".join(
            f"[{i}] {a['title']} — {a['content'][:200]}" for i, a in enumerate(articles[:50])
        )

        messages = [
            SystemMessage(
                content=(
                    "You are an AI/ML news analyst. For each article below, output a JSON array "
                    "where each element has: index (int), category (one of: LLM, Computer Vision, "
                    "Robotics, AI Policy, AI Startup, Research Paper, Industry News, Other), "
                    "and relevance_score (0.0-1.0, how important this is for AI practitioners). "
                    "Output ONLY valid JSON, no markdown fences."
                )
            ),
            HumanMessage(content=article_list),
        ]

        llm.invoke(messages)
        logger.info("analysis_complete", articles_analysed=min(len(articles), 50))

        # TODO: Parse JSON response, enrich articles with categories & scores
        # For now, pass through as-is
        return {"current_step": "analyzed"}

    except Exception as e:
        logger.error("analyze_error", error=str(e))
        return {"error_log": [f"Analysis error: {e}"], "current_step": "analyzed"}


# ═══════════════════════════════════════════════════════════════
# Summarization — generate newsletter-ready summaries
# ═══════════════════════════════════════════════════════════════
SUMMARIZE_SYSTEM_PROMPT = """You are a senior AI/ML journalist writing a weekly newsletter.
For each article provided, write a concise summary consisting of:
1. A compelling headline (max 80 chars)
2. A 2-3 sentence body capturing the key insight, why it matters, and any numbers/dates
3. Categorise as one of: LLM, Computer Vision, Robotics, AI Policy, AI Startup, Research, Industry

Output a JSON array with objects: {headline, body, category, source_url, credibility_score}.
Rank by importance — lead with the biggest story. Output ONLY valid JSON."""


def summarize_node(state: PipelineState) -> dict:
    """Generate polished summaries from deduplicated, scored articles."""
    articles = state.get("deduplicated_articles", [])
    feedback = state.get("feedback", "")
    if not articles:
        return {"error_log": ["Summarize: no articles to process"]}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model_summarizer,
            temperature=0.3,
            google_api_key=settings.google_api_key,
        )

        # Build article context (top articles by credibility)
        sorted_articles = sorted(articles, key=lambda a: a["credibility_score"], reverse=True)
        top_articles = sorted_articles[: settings.max_articles_per_run]

        article_context = "\n---\n".join(
            f"Title: {a['title']}\nSource: {a['source']}\nURL: {a['url']}\n"
            f"Credibility: {a['credibility_score']:.2f}\n"
            f"Content: {a['content'][:500]}"
            for a in top_articles
        )

        system_prompt = SUMMARIZE_SYSTEM_PROMPT
        if feedback:
            system_prompt += f"\n\nHuman feedback from previous draft: {feedback}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Here are today's articles:\n\n{article_context}"),
        ]

        response = llm.invoke(messages)
        logger.info(
            "summarization_complete",
            articles_input=len(top_articles),
            response_length=len(response.content),
        )

        # Parse JSON response into list[Summary]
        raw_text = response.content.strip()
        # Strip markdown fences if the model wraps in ```json ... ```
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

        parsed: list[dict] = json.loads(raw_text)
        summaries = [
            {
                "headline": item.get("headline", ""),
                "body": item.get("body", ""),
                "category": item.get("category", "Industry"),
                "source_urls": [item.get("source_url", "")],
                "credibility_score": float(item.get("credibility_score", 0.5)),
            }
            for item in parsed
            if isinstance(item, dict)
        ]
        logger.info("summaries_parsed", count=len(summaries))
        return {"summaries": summaries, "current_step": "summarized"}

    except Exception as e:
        logger.error("summarize_error", error=str(e))
        return {"error_log": [f"Summarization error: {e}"], "current_step": "summarized"}

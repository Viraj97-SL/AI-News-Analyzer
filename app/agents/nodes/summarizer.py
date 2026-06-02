"""
Summarizer nodes — deduplication, analysis, and LLM summarization.

Uses tiered model routing:
  - Flash for topic classification
  - Flash for summarization
  - Pro reserved for complex analysis (called from credibility node)
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
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


def _parse_json_tolerant(text: str) -> list[dict]:
    """
    Parse a JSON array, recovering from two common LLM failure modes:
    - Extra data: LLM appended explanation text after the closing bracket.
    - Truncation: LLM hit a token/output limit mid-stream.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # "Extra data" means valid JSON was found but trailing content follows it.
        # e.pos points to the start of the extra content; slice it off and retry.
        if "Extra data" in str(e):
            try:
                result = json.loads(text[: e.pos])
                logger.warning("json_extra_data_recovered", objects_recovered=len(result))
                return result
            except json.JSONDecodeError:
                pass

        # Truncation fallback: find the last complete object and close the array.
        last_close = text.rfind("},")
        if last_close == -1:
            last_close = text.rfind("}")
        if last_close != -1:
            truncated = text[: last_close + 1].strip()
            if not truncated.startswith("["):
                truncated = "[" + truncated
            truncated += "]"
            try:
                result = json.loads(truncated)
                logger.warning("json_truncation_recovered", objects_recovered=len(result))
                return result
            except json.JSONDecodeError:
                pass
        raise


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
_VALID_CATEGORIES = frozenset(
    ["LLM", "Computer Vision", "Robotics", "AI Policy", "AI Startup", "Research Paper",
     "Industry News", "Other"]
)


def analyze_node(state: PipelineState) -> dict:
    """
    Categorise articles into topics and rank by significance.
    Uses Gemini Flash (mid tier) for classification. Enriches each article
    in-place with 'category' and 'relevance_score' fields.
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

        batch = articles[:50]
        article_list = "\n".join(
            f"[{i}] {a['title']} — {a['content'][:200]}" for i, a in enumerate(batch)
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a news analyst. For each article below, output a JSON array "
                    "where each element has: index (int), category (one of: LLM, Computer Vision, "
                    "Robotics, AI Policy, AI Startup, Research Paper, Industry News, Other), "
                    "and relevance_score (0.0-1.0, how significant this is for the general public). "
                    "Prioritise stories with broad real-world impact over niche technical updates. "
                    "Output ONLY valid JSON, no markdown fences."
                )
            ),
            HumanMessage(content=article_list),
        ]

        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            raw_text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            ).strip()
        else:
            raw_text = content.strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

        parsed: list[dict] = _parse_json_tolerant(raw_text)
        enriched = list(articles)  # shallow copy so we don't mutate state directly
        for item in parsed:
            idx = item.get("index")
            if not isinstance(idx, int) or idx >= len(enriched):
                continue
            cat = item.get("category", "Other")
            enriched[idx] = {
                **enriched[idx],
                "category": cat if cat in _VALID_CATEGORIES else "Other",
                "relevance_score": float(item.get("relevance_score", 0.5)),
            }

        logger.info(
            "analysis_complete",
            articles_analysed=len(batch),
            enriched=len(parsed),
        )
        return {"deduplicated_articles": enriched, "current_step": "analyzed"}

    except Exception as e:
        logger.error("analyze_error", error=str(e))
        return {"error_log": [f"Analysis error: {e}"], "current_step": "analyzed"}


# ═══════════════════════════════════════════════════════════════
# Ranking — composite score for article prioritisation
# ═══════════════════════════════════════════════════════════════
def _rank_score(article: dict) -> float:
    """
    Composite ranking: 35% credibility + 40% relevance + 25% recency.

    Recency decays linearly from 1.0 (today) to 0.0 (7+ days old).
    Falls back gracefully when optional fields are absent.
    """
    credibility = float(article.get("credibility_score", 0.5))
    relevance = float(article.get("relevance_score", 0.5))

    recency = 0.5  # neutral default
    pub_raw = article.get("published_at", "")
    if pub_raw:
        try:
            pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - pub_dt).total_seconds() / 86400
            recency = max(0.0, 1.0 - age_days / 7.0)
        except ValueError:
            pass

    return 0.35 * credibility + 0.40 * relevance + 0.25 * recency


# ═══════════════════════════════════════════════════════════════
# Summarization — generate newsletter-ready summaries
# ═══════════════════════════════════════════════════════════════
SUMMARIZE_SYSTEM_PROMPT = """You are a news journalist writing a weekly AI briefing for a general audience.
For each story provided, write a clear, jargon-free summary:
1. headline: Punchy, plain-English headline (max 80 chars). No acronyms without explanation.
2. body: 3-5 sentences covering what happened, key numbers or dates, why it matters in everyday terms, \
and what changes as a result. Avoid terms like "LLM", "RLHF", or "inference" without a brief explanation.
3. category: one of LLM, Computer Vision, Robotics, AI Policy, AI Startup, Research, Industry
4. source_urls: JSON array of all article URLs that informed this summary (can be multiple)
5. outlet_names: JSON array of up to 3 publication names (e.g. ["NYT", "Bloomberg", "Reuters"]) \
extracted from the outlet field in the input articles
6. bias_notes: If multiple outlets covered the same story with notably different angles or emphasis, \
note the key difference in one sentence. Use "" if single source or angles are similar.
7. credibility_score: 0.0–1.0 based on source quality

Output a JSON array of objects with exactly those 7 keys.
Rank by public significance — lead with the story that affects the most people.
Output ONLY valid JSON, no markdown."""


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

        # Sort by composite score: credibility + relevance + recency
        sorted_articles = sorted(articles, key=_rank_score, reverse=True)
        top_articles = sorted_articles[: settings.max_articles_per_run]

        def _outlet_label(article: dict) -> str:
            """Return a human-readable outlet name from the article URL."""
            from urllib.parse import urlparse
            try:
                domain = urlparse(article.get("url", "")).netloc.lower().replace("www.", "")
                # Use feed name embedded in rss source tag when available
                src = article.get("source", "")
                if src.startswith("rss:"):
                    return src.replace("rss:", "").replace("_", " ").title()
                parts = domain.split(".")
                if len(parts) >= 2:
                    return parts[-2].title()
                return domain or "Unknown"
            except Exception:
                return "Unknown"

        article_context = "\n---\n".join(
            f"Title: {a['title']}\nOutlet: {_outlet_label(a)}\nURL: {a['url']}\n"
            f"Content: {a['content'][:800]}"
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
        content = response.content
        if isinstance(content, list):
            raw_text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            ).strip()
        else:
            raw_text = content.strip()

        logger.info(
            "summarization_complete",
            articles_input=len(top_articles),
            response_length=len(raw_text),
        )

        # Strip markdown fences if the model wraps in ```json ... ```
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

        parsed: list[dict] = _parse_json_tolerant(raw_text)
        summaries = [
            {
                "headline": item.get("headline", ""),
                "body": item.get("body", ""),
                "category": item.get("category", "Industry"),
                # Accept both old single-url and new array format
                "source_urls": (
                    item["source_urls"] if isinstance(item.get("source_urls"), list)
                    else [item.get("source_url") or item.get("source_urls") or ""]
                ),
                "outlet_names": item.get("outlet_names") or [],
                "bias_notes": item.get("bias_notes") or "",
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


# ═══════════════════════════════════════════════════════════════
# Story Clustering — group articles by topic for cross-outlet analysis
# ═══════════════════════════════════════════════════════════════
_CLUSTER_SYSTEM_PROMPT = """You are a news editor. Group the articles below into story clusters —
each cluster represents a single real-world news event or topic covered by multiple outlets.

For each article assign a cluster_id (short slug like "openai-gpt5" or "eu-ai-act-fine").
Articles about the same event from different outlets share the same cluster_id.
Unrelated articles each get a unique cluster_id.

Output a JSON array: [{index: int, cluster_id: str}].
Output ONLY valid JSON."""


def cluster_stories_node(state) -> dict:
    """
    Assign story cluster IDs to deduplicated articles using LLM grouping.
    Articles about the same event across different outlets share a cluster_id.
    Enriches each article with a 'story_cluster_id' field.
    """
    articles = state.get("deduplicated_articles", [])
    if len(articles) < 2:
        return {}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model_classifier,
            temperature=0,
            google_api_key=settings.google_api_key,
        )

        batch = articles[:60]
        article_list = "\n".join(
            f"[{i}] {a['title']}" for i, a in enumerate(batch)
        )

        messages = [
            SystemMessage(content=_CLUSTER_SYSTEM_PROMPT),
            HumanMessage(content=article_list),
        ]

        response = llm.invoke(messages)
        content = response.content
        raw_text = content if isinstance(content, str) else "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

        parsed: list[dict] = _parse_json_tolerant(raw_text)
        enriched = list(articles)
        for item in parsed:
            idx = item.get("index")
            if isinstance(idx, int) and idx < len(enriched):
                enriched[idx] = {**enriched[idx], "story_cluster_id": item.get("cluster_id", "")}

        logger.info("clustering_complete", clusters=len({a.get("story_cluster_id") for a in enriched}))
        return {"deduplicated_articles": enriched, "current_step": "clustered"}

    except Exception as e:
        logger.warning("cluster_error", error=str(e))
        return {}

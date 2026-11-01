"""
Research-only paper sourcing (decision D1 — the shared `scraper.py` is used by the
news pipeline and is intentionally left untouched).

Sources (all fault-tolerant; a failing source returns [] plus an error_log entry):
  - Hugging Face Daily Papers — community-voted feed that surfaces big-lab
    releases within hours, with organisation, upvotes and GitHub stars (D5).
  - arXiv — newest papers across cs.AI/LG/CL/CV plus cs.MA (multi-agent) and
    cs.IR (retrieval / RAG); the author `comment` field is kept as an
    affiliation hint ("Work done at Google DeepMind").

`merge_research_candidates` dedupes by arXiv id across sources, merges their
signals, and enriches each paper with lab + trend tags for the ranker.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.lab_registry import detect_lab
from app.agents.nodes.trends import tag_trends
from app.core.logging import get_logger

logger = get_logger(__name__)

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
HF_LIMIT = 100
HF_LOOKBACK_DAYS = 10  # HF lists papers for ~2 weeks; keep the recent, still-hot part
ARXIV_LOOKBACK_DAYS = 7
ARXIV_MAX_RESULTS = 150
# Trend-focused abstract terms: a weekly window holds thousands of cs.* papers, so an
# unfiltered "newest N" query only ever sees the last hour or two. HF Daily Papers is
# the primary "what's hot" source; arXiv fills in trend papers it hasn't picked up yet.
ARXIV_TREND_TERMS: tuple[str, ...] = (
    '"model context protocol"',
    '"tool calling"',
    '"retrieval-augmented"',
    "RAG",
    '"multi-agent"',
    '"LLM agent"',
    "agentic",
    '"test-time compute"',
    '"reasoning model"',
    '"mixture of experts"',
    '"world model"',
    '"mechanistic interpretability"',
    '"long context"',
)
RESEARCH_ARXIV_CATEGORIES: tuple[str, ...] = ("cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.MA", "cs.IR")
_HTTP_TIMEOUT_S = 20
_USER_AGENT = "ai-news-summarizer-research/1.0"

# Community-attention bonus (ranker): log-scaled so one viral paper can't dominate.
MAX_ATTENTION_BONUS = 2.0
_UPVOTE_BONUS_CAP = 1.5
_STAR_BONUS_CAP = 0.5

_SOURCE_PRIORITY = {"manual": 0, "hf_daily": 1, "arxiv": 2}
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|huggingface\.co/papers/|^)(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?:$|[/?#])"
)


def _now() -> datetime:
    return datetime.now(UTC)


def canonical_arxiv_id(url: str) -> str | None:
    match = _ARXIV_ID_RE.search(url or "")
    return match.group(1) if match else None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ── Hugging Face Daily Papers ────────────────────────────────────────────────


def _hf_item_to_article(item: dict[str, Any]) -> dict[str, Any] | None:
    paper = item.get("paper") or {}
    pid = paper.get("id")
    title = paper.get("title") or item.get("title")
    if not pid or not title:
        return None
    org = item.get("organization") or paper.get("organization") or {}
    keywords = [k for k in paper.get("ai_keywords") or [] if isinstance(k, str)]
    summary = paper.get("summary") or item.get("summary") or ""
    content = f"{summary}\n\nKeywords: {', '.join(keywords)}" if keywords else summary
    return {
        "title": title.strip(),
        "url": f"https://arxiv.org/abs/{pid}",
        "source": "hf_daily",
        "content": content,
        "published_at": paper.get("publishedAt") or item.get("publishedAt") or "",
        "credibility_score": 0.85,
        "upvotes": int(paper.get("upvotes") or 0),
        "github_stars": int(paper.get("githubStars") or 0),
        "github_repo": paper.get("githubRepo") or "",
        "affiliation_hint": org.get("fullname") or org.get("name") or "",
    }


def parse_hf_daily_papers(items: list[Any], now: datetime) -> list[dict[str, Any]]:
    """Map raw API items to article dicts, dropping malformed and stale entries."""
    cutoff = now - timedelta(days=HF_LOOKBACK_DAYS)
    articles: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        article = _hf_item_to_article(item)
        if article is None:
            continue
        published = _parse_ts(article["published_at"])
        if published is not None and published < cutoff:
            continue
        articles.append(article)
    return articles


def scrape_hf_daily_papers_node(state: PipelineState) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}) as client:
            response = client.get(HF_DAILY_PAPERS_URL, params={"limit": HF_LIMIT})
            response.raise_for_status()
            items = response.json()
        if not isinstance(items, list):
            raise ValueError(f"unexpected payload type {type(items).__name__}")
        articles = parse_hf_daily_papers(items, now=_now())
        logger.info("hf_daily_papers_scraped", count=len(articles))
        return {"raw_articles": articles}
    except Exception as e:
        logger.warning("hf_daily_papers_failed", error=str(e))
        return {"raw_articles": [], "error_log": [f"HF Daily Papers error: {e}"]}


# ── arXiv ────────────────────────────────────────────────────────────────────


def research_arxiv_query() -> str:
    cats = " OR ".join(f"cat:{c}" for c in RESEARCH_ARXIV_CATEGORIES)
    terms = " OR ".join(f"abs:{t}" for t in ARXIV_TREND_TERMS)
    return f"({cats}) AND ({terms})"


def scrape_research_arxiv_node(state: PipelineState) -> dict[str, Any]:
    try:
        import arxiv

        search = arxiv.Search(
            query=research_arxiv_query(),
            max_results=ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        client = arxiv.Client(delay_seconds=5, num_retries=1)
        cutoff = _now() - timedelta(days=ARXIV_LOOKBACK_DAYS)
        articles: list[dict[str, Any]] = []
        for result in client.results(search):
            published = (
                result.published
                if result.published.tzinfo
                else result.published.replace(tzinfo=UTC)
            )
            if published < cutoff:
                continue
            articles.append(
                {
                    "title": result.title,
                    "url": result.entry_id,
                    "source": "arxiv",
                    "content": result.summary,
                    "published_at": published.isoformat(),
                    "credibility_score": 0.8,
                    "affiliation_hint": " ".join(
                        filter(None, [result.comment, result.journal_ref])
                    ),
                }
            )
        logger.info("research_arxiv_scraped", count=len(articles))
        return {"raw_articles": articles}
    except Exception as e:
        logger.warning("research_arxiv_failed", error=str(e))
        return {"raw_articles": [], "error_log": [f"Research arXiv error: {e}"]}


# ── merge & enrich ───────────────────────────────────────────────────────────


def attention_bonus(article: dict[str, Any]) -> float:
    upvotes = max(0, int(article.get("upvotes") or 0))
    stars = max(0, int(article.get("github_stars") or 0))
    upvote_part = min(_UPVOTE_BONUS_CAP, math.log2(1 + upvotes) / 4)  # 15 → 1.0, 63 → 1.5
    star_part = min(_STAR_BONUS_CAP, math.log10(1 + stars) / 6)  # 1k → 0.5
    return round(min(MAX_ATTENTION_BONUS, upvote_part + star_part), 3)


def _merge_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(group, key=lambda a: _SOURCE_PRIORITY.get(a.get("source", ""), 9))
    primary = ordered[0]
    hints = [a.get("affiliation_hint") or "" for a in ordered]
    # HF `organization` / manual hints are clean org names → loose matching.
    # arXiv comments are free text ("built on Qwen", "github.com/google/...") → strict.
    org_hints = [a.get("affiliation_hint") or "" for a in ordered if a.get("source") != "arxiv"]
    free_text_hints = [
        a.get("affiliation_hint") or "" for a in ordered if a.get("source") == "arxiv"
    ]
    lab = detect_lab(*org_hints) or detect_lab(*free_text_hints, strict=True)
    text = " ".join(f"{a.get('title') or ''} {a.get('content') or ''}" for a in ordered)
    longest_content = max((a.get("content") or "" for a in ordered), key=len)
    return {
        **primary,
        "content": longest_content,
        "upvotes": max(int(a.get("upvotes") or 0) for a in ordered),
        "github_stars": max(int(a.get("github_stars") or 0) for a in ordered),
        "github_repo": next((a["github_repo"] for a in ordered if a.get("github_repo")), ""),
        "affiliation_hint": " | ".join(h for h in hints if h),
        "lab": lab.name if lab else None,
        "lab_tier": lab.tier if lab else None,
        "trend_tags": tag_trends(text),
        "sources": sorted({a.get("source", "") for a in ordered}),
    }


def merge_research_candidates(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by arXiv id (non-arXiv items by URL), merge signals, add lab + trend tags.

    Returns new dicts; inputs are not mutated. Order follows first appearance.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        url = article.get("url") or ""
        if not url:
            logger.warning(
                "research_candidate_without_url", title=(article.get("title") or "")[:80]
            )
            continue
        key = canonical_arxiv_id(url) or url
        groups.setdefault(key, []).append(article)
    return [_merge_group(group) for group in groups.values()]

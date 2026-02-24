"""
Scraper nodes — fan-out to 4 sources in parallel, then merge.

Each scraper returns {"raw_articles": [...]} which the Annotated reducer
on PipelineState.raw_articles merges automatically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from app.agents.state import NewsArticle, PipelineState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════
# Tier 1: Tavily Search
# ═══════════════════════════════════════════════════════════════
def scrape_tavily_node(state: PipelineState) -> dict:
    """Search AI/ML news via Tavily's LLM-optimised search API."""
    if not settings.tavily_api_key:
        logger.warning("tavily_skipped", reason="no API key configured")
        return {"raw_articles": [], "error_log": ["Tavily: no API key"]}

    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(
            max_results=10,
            topic="news",
            search_depth="advanced",
            time_range="week",
            include_domains=[
                "techcrunch.com",
                "venturebeat.com",
                "theverge.com",
                "wired.com",
                "arstechnica.com",
                "thenewstack.io",
            ],
        )

        queries = [
            "artificial intelligence machine learning news this week",
            "large language model LLM breakthroughs 2025",
            "AI startup funding and product launches",
        ]

        articles: list[NewsArticle] = []
        for query in queries:
            results = tool.invoke(query)
            if isinstance(results, list):
                for r in results:
                    articles.append(
                        NewsArticle(
                            title=r.get("title", "Untitled"),
                            url=r.get("url", ""),
                            source="tavily",
                            content=r.get("content", ""),
                            published_at=r.get("published_date", datetime.now(UTC).isoformat()),
                            credibility_score=0.0,
                        )
                    )

        logger.info("tavily_scraped", article_count=len(articles))
        return {"raw_articles": articles}

    except Exception as e:
        logger.error("tavily_error", error=str(e))
        return {"raw_articles": [], "error_log": [f"Tavily error: {e}"]}


# ═══════════════════════════════════════════════════════════════
# Tier 2: RSS Feed Aggregation
# ═══════════════════════════════════════════════════════════════
RSS_FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ("Google AI Blog", "https://blog.google/technology/ai/rss/"),
    (
        "Google News AI",
        "https://news.google.com/rss/search?q=artificial+intelligence+machine+learning&hl=en-US&gl=US&ceid=US:en",
    ),
]


def scrape_rss_node(state: PipelineState) -> dict:
    """Parse curated RSS feeds for AI/ML articles."""
    try:
        import feedparser

        articles: list[NewsArticle] = []
        cutoff = datetime.now(UTC) - timedelta(days=7)

        for feed_name, feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:  # cap per feed
                    published = entry.get("published_parsed")
                    if published:
                        pub_dt = datetime(*published[:6], tzinfo=UTC)
                        if pub_dt < cutoff:
                            continue

                    articles.append(
                        NewsArticle(
                            title=entry.get("title", "Untitled"),
                            url=entry.get("link", ""),
                            source=f"rss:{feed_name.lower().replace(' ', '_')}",
                            content=entry.get("summary", entry.get("description", "")),
                            published_at=entry.get("published", datetime.now(UTC).isoformat()),
                            credibility_score=0.0,
                        )
                    )
            except Exception as e:
                logger.warning("rss_feed_error", feed=feed_name, error=str(e))

        logger.info("rss_scraped", article_count=len(articles), feeds_checked=len(RSS_FEEDS))
        return {"raw_articles": articles}

    except Exception as e:
        logger.error("rss_error", error=str(e))
        return {"raw_articles": [], "error_log": [f"RSS error: {e}"]}


# ═══════════════════════════════════════════════════════════════
# Tier 3: ArXiv Research Papers
# ═══════════════════════════════════════════════════════════════
def scrape_arxiv_node(state: PipelineState) -> dict:
    """Fetch latest AI/ML research papers from ArXiv."""
    try:
        import arxiv

        search = arxiv.Search(
            query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV",
            max_results=20,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        client = arxiv.Client(delay_seconds=3)  # respect rate limits
        articles: list[NewsArticle] = []

        for result in client.results(search):
            articles.append(
                NewsArticle(
                    title=result.title,
                    url=result.entry_id,
                    source="arxiv",
                    content=result.summary,
                    published_at=result.published.isoformat(),
                    credibility_score=0.8,  # arxiv papers are peer-adjacent
                )
            )

        logger.info("arxiv_scraped", article_count=len(articles))
        return {"raw_articles": articles}

    except Exception as e:
        logger.error("arxiv_error", error=str(e))
        return {"raw_articles": [], "error_log": [f"ArXiv error: {e}"]}


# ═══════════════════════════════════════════════════════════════
# Tier 4: Serper.dev (Google News fallback)
# ═══════════════════════════════════════════════════════════════
def scrape_serper_node(state: PipelineState) -> dict:
    """Search Google News via Serper as a fallback/gap-filler."""
    if not settings.serper_api_key:
        logger.info("serper_skipped", reason="no API key configured")
        return {"raw_articles": []}

    try:
        articles: list[NewsArticle] = []
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": settings.serper_api_key},
                json={"q": "AI machine learning news", "num": 10},
            )
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("news", []):
            articles.append(
                NewsArticle(
                    title=item.get("title", "Untitled"),
                    url=item.get("link", ""),
                    source="serper",
                    content=item.get("snippet", ""),
                    published_at=item.get("date", datetime.now(UTC).isoformat()),
                    credibility_score=0.0,
                )
            )

        logger.info("serper_scraped", article_count=len(articles))
        return {"raw_articles": articles}

    except Exception as e:
        logger.error("serper_error", error=str(e))
        return {"raw_articles": [], "error_log": [f"Serper error: {e}"]}


# ═══════════════════════════════════════════════════════════════
# Merge node — no-op, the Annotated reducer handles merging
# ═══════════════════════════════════════════════════════════════
def merge_results_node(state: PipelineState) -> dict:
    """Log merge stats. Actual merging is handled by the state reducer."""
    total = len(state.get("raw_articles", []))
    sources = set(a["source"] for a in state.get("raw_articles", []))
    logger.info("articles_merged", total=total, sources=list(sources))
    return {"current_step": "merged"}

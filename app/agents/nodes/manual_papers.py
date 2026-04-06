"""
Manual papers node — loads user-curated papers into the research pipeline.

Two sources are checked:
  1. state['manual_paper_url']  — single URL passed when triggering the API.
  2. data/manual_papers.json    — persistent reading list; queue entries are
                                   processed and moved to archive after each run.

All loaded papers are returned as raw_articles with source='manual' so the
rank_papers_node gives them a strong priority boost over scraped ArXiv papers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

_MANUAL_PAPERS_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "manual_papers.json"
)


def _fetch_arxiv_paper(url_or_id: str) -> dict | None:
    """
    Fetch ArXiv paper metadata and return a NewsArticle-compatible dict.
    Accepts:
      - https://arxiv.org/abs/2301.07685
      - https://arxiv.org/abs/2301.07685v2
      - 2301.07685
    """
    try:
        import arxiv  # type: ignore[import]

        # Extract bare ID from any format
        raw = url_or_id.strip().rstrip("/").split("/")[-1]
        # Strip version suffix (v1, v2, ...)
        import re
        arxiv_id = re.sub(r"v\d+$", "", raw)

        client = arxiv.Client(delay_seconds=3, num_retries=2)
        results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
        if not results:
            logger.warning("manual_paper_not_found_on_arxiv", id=arxiv_id)
            return None

        r = results[0]
        return {
            "title": r.title,
            "url": r.entry_id,
            "source": "manual",
            "content": r.summary,
            "published_at": r.published.isoformat(),
            "credibility_score": 0.95,  # user-curated papers get max trust
        }

    except Exception as e:
        logger.warning("manual_paper_fetch_failed", url=url_or_id, error=str(e))
        return None


def load_manual_papers_node(state: "PipelineState") -> dict:
    """
    Load manually curated papers and add them to raw_articles.

    Priority order:
      1. manual_paper_url from state (API-triggered, one-shot override).
      2. Queue in data/manual_papers.json (reading list).

    Processed queue entries are moved to 'archive' so they are not re-run.
    """
    articles: list[dict] = []

    # ── 1. State-level override (API trigger) ────────────────────────────
    manual_url = state.get("manual_paper_url", "")
    if manual_url:
        logger.info("loading_manual_paper_from_state", url=manual_url)
        paper = _fetch_arxiv_paper(manual_url)
        if paper:
            articles.append(paper)
            logger.info("manual_state_paper_loaded", title=paper["title"])

    # ── 2. Reading list file ─────────────────────────────────────────────
    if _MANUAL_PAPERS_PATH.exists():
        try:
            data = json.loads(_MANUAL_PAPERS_PATH.read_text(encoding="utf-8"))
            queue: list[dict] = data.get("queue", [])

            if queue:
                logger.info("manual_queue_found", count=len(queue))
                processed: list[dict] = []

                for entry in queue:
                    url = entry.get("url", "").strip()
                    if not url:
                        continue
                    paper = _fetch_arxiv_paper(url)
                    if paper:
                        # Prepend user note to abstract for context during analysis
                        if entry.get("note"):
                            paper["content"] = (
                                f"[Curator note: {entry['note']}]\n\n{paper['content']}"
                            )
                        articles.append(paper)
                        logger.info("queued_paper_loaded", title=paper["title"])
                    processed.append({**entry, "processed_at": datetime.now(UTC).isoformat()})

                # Archive processed entries so they are not re-processed
                data["queue"] = []
                data.setdefault("archive", []).extend(processed)
                _MANUAL_PAPERS_PATH.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                logger.info("manual_queue_archived", count=len(processed))

        except Exception as e:
            logger.warning("manual_papers_file_error", error=str(e))

    logger.info("manual_papers_total_loaded", count=len(articles))
    return {"raw_articles": articles, "current_step": "manual_papers_loaded"}

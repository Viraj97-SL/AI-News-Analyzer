"""
Manual papers node — loads user-curated papers into the research pipeline.

Three sources are checked (in priority order):
  1. state['manual_paper_url']  — single URL passed when triggering the API.
  2. data/manual_papers.json queue — persistent reading list.
  3. Auto-rotation from archive — randomly injects a landmark paper (~33% of runs)
     when queue is empty and no API override is present.

All loaded papers are returned as raw_articles with source='manual' so the
rank_papers_node gives them a strong priority boost over scraped ArXiv papers.
"""

from __future__ import annotations

import json
import random
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


_MAX_QUEUE_PER_RUN = 1  # Process at most 1 queued paper per run
_CLASSIC_INJECTION_PROBABILITY = 0.33  # ~1 in 3 runs when queue is empty
_RECENTLY_FEATURED_MAX = 12  # avoid re-featuring the same paper too soon


def _pick_classic_paper(data: dict) -> dict | None:
    """
    Randomly pick one landmark paper from the archive that hasn't been
    featured recently, then record it in recently_featured.
    Returns a queue-style entry dict, or None if archive is empty.
    """
    archive: list[dict] = data.get("archive", [])
    if not archive:
        return None

    recently_featured: list[str] = data.get("recently_featured", [])
    recent_urls = set(recently_featured[-_RECENTLY_FEATURED_MAX:])

    # Build candidate pool excluding recently featured entries
    candidates = [e for e in archive if e.get("url", "") not in recent_urls]
    if not candidates:
        # All have been featured recently — reset and pick from full archive
        candidates = archive

    chosen = random.choice(candidates)

    # Track this pick so we don't repeat it soon
    recently_featured.append(chosen.get("url", ""))
    data["recently_featured"] = recently_featured[-_RECENTLY_FEATURED_MAX:]

    return chosen


def load_manual_papers_node(state: "PipelineState") -> dict:
    """
    Load manually curated papers and add them to raw_articles.

    Priority order:
      1. manual_paper_url from state (API-triggered, one-shot override).
      2. Queue in data/manual_papers.json (reading list).
      3. Auto-rotation: if queue is empty and no override, randomly (~33%)
         inject a landmark paper from the archive as a Classic Paper Series run.

    Processed queue entries are moved to 'archive' so they are not re-run.
    is_classic_paper=True is set in state when source 3 triggers.

    Note: On Railway (ephemeral containers) file writes don't persist between
    deployments. Keep the queue short (0-1 entries) and prefer the API trigger.
    """
    articles: list[dict] = []
    is_classic_paper = False

    # ── 1. State-level override (API trigger) ────────────────────────────
    manual_url = state.get("manual_paper_url", "")
    if manual_url:
        logger.info("loading_manual_paper_from_state", url=manual_url)
        paper = _fetch_arxiv_paper(manual_url)
        if paper:
            articles.append(paper)
            logger.info("manual_state_paper_loaded", title=paper["title"])

    # ── 2. Reading list file + 3. Classic paper auto-rotation ────────────
    if _MANUAL_PAPERS_PATH.exists():
        try:
            data = json.loads(_MANUAL_PAPERS_PATH.read_text(encoding="utf-8"))
            queue: list[dict] = data.get("queue", [])

            batch = queue[:_MAX_QUEUE_PER_RUN]

            if batch:
                logger.info("manual_queue_found", total=len(queue), processing=len(batch))
                processed: list[dict] = []

                for entry in batch:
                    url = entry.get("url", "").strip()
                    if not url:
                        continue
                    paper = _fetch_arxiv_paper(url)
                    if paper:
                        if entry.get("note"):
                            paper["content"] = (
                                f"[Curator note: {entry['note']}]\n\n{paper['content']}"
                            )
                        articles.append(paper)
                        logger.info("queued_paper_loaded", title=paper["title"])
                    processed.append({**entry, "processed_at": datetime.now(UTC).isoformat()})

                data["queue"] = queue[_MAX_QUEUE_PER_RUN:]
                data.setdefault("archive", []).extend(processed)
                _MANUAL_PAPERS_PATH.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                logger.info("manual_queue_partially_archived", archived=len(processed), remaining=len(data["queue"]))

            elif not manual_url and random.random() < _CLASSIC_INJECTION_PROBABILITY:
                # Queue is empty and no API override — randomly inject a landmark paper
                classic_entry = _pick_classic_paper(data)
                if classic_entry:
                    url = classic_entry.get("url", "").strip()
                    paper = _fetch_arxiv_paper(url) if url else None
                    if paper:
                        note = classic_entry.get("note", "")
                        if note:
                            paper["content"] = (
                                f"[Classic Paper — {note}]\n\n{paper['content']}"
                            )
                        articles.append(paper)
                        is_classic_paper = True
                        logger.info(
                            "classic_paper_auto_injected",
                            title=paper["title"],
                            url=url,
                        )
                        # Persist the updated recently_featured list
                        _MANUAL_PAPERS_PATH.write_text(
                            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                        )

        except Exception as e:
            logger.warning("manual_papers_file_error", error=str(e))

    logger.info("manual_papers_total_loaded", count=len(articles), is_classic=is_classic_paper)
    return {
        "raw_articles": articles,
        "is_classic_paper": is_classic_paper,
        "current_step": "manual_papers_loaded",
    }

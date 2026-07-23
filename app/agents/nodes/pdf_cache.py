"""
Disk-backed ArXiv PDF cache.

Both figure extraction (`architecture_diagram.py`) and full-text extraction
(`full_text.py`) need the same PDF. Without a shared cache each would fetch
it independently on every run; this makes the second fetch within a run (or
across runs, for classic/reprocessed papers) a disk read instead of a
network call.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("./data/pdf_cache")


def fetch_pdf_bytes(
    paper_url: str,
    arxiv_id: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout: float = 30.0,
) -> bytes | None:
    """Return the paper's PDF bytes, downloading and caching by `arxiv_id` on first use."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{arxiv_id}.pdf"

    if cache_path.exists():
        logger.info("pdf_cache_hit", arxiv_id=arxiv_id)
        return cache_path.read_bytes()

    pdf_url = paper_url.replace("/abs/", "/pdf/")
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(pdf_url, headers={"User-Agent": "research-analyzer/1.0"})
            resp.raise_for_status()
    except Exception as e:
        logger.warning("pdf_download_failed", arxiv_id=arxiv_id, error=str(e))
        return None

    cache_path.write_bytes(resp.content)
    logger.info("pdf_cache_stored", arxiv_id=arxiv_id, bytes=len(resp.content))
    return resp.content

"""Shared ArXiv URL helpers used by figure extraction and full-text fetching."""

from __future__ import annotations

import re


def extract_arxiv_id(paper_url: str) -> str:
    """Return the bare ArXiv ID (e.g. '2301.07685') from any URL format."""
    raw = paper_url.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", raw)

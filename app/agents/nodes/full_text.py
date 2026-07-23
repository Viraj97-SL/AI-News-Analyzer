"""
Full-text extraction node — downloads the chosen paper's PDF and pulls out
the Results / Ablation / Experiment Setup sections specifically.

Abstract-only analysis is the single largest cause of thin carousel
content: the LLM has no choice but to write "not available in the
abstract" for anything past the summary paragraph. This node runs right
after paper selection, before `deep_analysis_node`, so those three
sections are available to the analysis prompt alongside the abstract.

Falls back to abstract-only mode (existing `paper["content"]`, untouched)
and flags `analysis_confidence: "low"` whenever the PDF can't be fetched,
parsed, or doesn't contain any of the target section headings.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.arxiv_utils import extract_arxiv_id
from app.agents.nodes.pdf_cache import fetch_pdf_bytes
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_SECTION_HEADING_PATTERNS: dict[str, re.Pattern[str]] = {
    "experiment_setup": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(experimental setup|experiment setup|implementation details|"
        r"setup(?:\s+and\s+datasets)?|datasets? and (?:baselines|metrics))\s*$"
    ),
    "results": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(experiments? and results|main results|quantitative results|"
        r"experimental results|results)\s*$"
    ),
    "ablation": re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(ablation stud(?:y|ies)|ablations?|ablation experiments)\s*$"
    ),
}

# A generic "next numbered/titled section" boundary — bounds each extracted
# section so it doesn't run past the start of the following one.
_NEXT_HEADING_RE = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)*\.?\s+)[A-Z][A-Za-z0-9 ,\-:]{2,60}\s*$")

_EMPTY_RESULT = {"full_text_available": False, "analysis_confidence": "low", "paper_sections": {}}


def _extract_section(raw_text: str, pattern: re.Pattern[str], max_chars: int) -> str:
    match = pattern.search(raw_text)
    if not match:
        return ""
    start = match.end()
    next_heading = _NEXT_HEADING_RE.search(raw_text, pos=start)
    end = next_heading.start() if next_heading else len(raw_text)
    return raw_text[start:end].strip()[:max_chars]


def fetch_full_text_node(state: "PipelineState") -> dict:
    """Download the chosen paper's PDF and extract Results/Ablation/Setup sections."""
    paper = state.get("chosen_research_paper", {})
    paper_url = paper.get("url", "")
    if not paper_url:
        return dict(_EMPTY_RESULT)

    arxiv_id = extract_arxiv_id(paper_url)
    pdf_bytes = fetch_pdf_bytes(paper_url, arxiv_id)
    if not pdf_bytes:
        logger.info("full_text_unavailable_pdf_fetch_failed", arxiv_id=arxiv_id)
        return dict(_EMPTY_RESULT)

    try:
        import fitz  # type: ignore[import]

        doc = fitz.Document(stream=pdf_bytes, filetype="pdf")  # type: ignore[call-arg]
        raw_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        logger.warning("full_text_pdf_parse_failed", arxiv_id=arxiv_id, error=str(e))
        return dict(_EMPTY_RESULT)

    sections = {
        name: _extract_section(raw_text, pattern, settings.full_text_section_max_chars)
        for name, pattern in _SECTION_HEADING_PATTERNS.items()
    }
    sections = {name: text for name, text in sections.items() if text}

    if not sections:
        logger.info("full_text_no_target_sections_found", arxiv_id=arxiv_id)
        return dict(_EMPTY_RESULT)

    logger.info(
        "full_text_extracted",
        arxiv_id=arxiv_id,
        sections_found=list(sections.keys()),
        total_chars=sum(len(text) for text in sections.values()),
    )
    return {
        "full_text_available": True,
        "analysis_confidence": "high",
        "paper_sections": sections,
    }

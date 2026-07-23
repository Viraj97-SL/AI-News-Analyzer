"""
Character-budget enforcement for carousel-rendered text fields.

The carousel template clips overflowing text with CSS `-webkit-line-clamp`,
which silently ends mid-word ("...") once a field runs past its slide's
line count. That's a rendering-layer symptom of a generation-layer problem:
the LLM was never told how much space it had.

This module fits each field to its slide's character budget *before*
render: first by asking the LLM to rewrite within budget (up to
`carousel_text_max_retries` attempts), then — only if that still fails —
by trimming at a sentence/word boundary and logging a structured error.
The CSS line-clamp remains in the template only as a last-resort visual
safety net; it should rarely, if ever, fire once this runs first.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Fields rendered behind a hard `-webkit-line-clamp` in research_carousel_slide.html.
# Budget class is the tightest slide context each field appears in.
BODY_FIELDS: tuple[str, ...] = (
    "core_problem", "methodology", "experiment_setup", "ablation_highlights", "breakthroughs",
)
SHORT_FIELDS: tuple[str, ...] = (
    "technical_innovation", "ecosystem_impact", "expert_interpretation", "limitations",
)


def _budget_for(field: str) -> int:
    if field in SHORT_FIELDS:
        return settings.carousel_short_char_budget
    return settings.carousel_body_char_budget


def _regenerate_within_budget(field_name: str, current_text: str, budget: int, paper_title: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=settings.google_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the given text to fit within {budget} characters (hard limit). "
         "Preserve the most important facts and numbers. End on a complete sentence — "
         "never cut off mid-word or mid-sentence, and never append '...'. "
         "Return ONLY the rewritten text, no explanation."),
        ("user", "Paper: {title}\n\nField: {field_name}\n\nOriginal text:\n{text}"),
    ])
    resp = (prompt | llm).invoke({
        "budget": budget, "title": paper_title, "field_name": field_name, "text": current_text,
    }).content
    rewritten = (
        "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in resp).strip()
        if isinstance(resp, list) else resp.strip()
    )
    return rewritten or current_text


def _trim_to_boundary(text: str, budget: int) -> str:
    """Last-resort trim: cut at the last sentence end, else the last word, within budget.
    Never cuts mid-word and never appends an ellipsis."""
    if len(text) <= budget:
        return text
    window = text[:budget]
    for sep in (". ", "! ", "? "):
        idx = window.rfind(sep)
        if idx != -1 and idx > budget * 0.4:
            return window[: idx + 1].strip()
    idx = window.rfind(" ")
    return (window[:idx] if idx != -1 else window).strip()


def enforce_char_budgets(analysis_dict: dict, paper_title: str) -> dict:
    """Return a copy of `analysis_dict` with every carousel-rendered text field
    fit to its slide's character budget. See module docstring for the strategy."""
    result = dict(analysis_dict)

    for field in (*BODY_FIELDS, *SHORT_FIELDS):
        text = result.get(field, "")
        if not isinstance(text, str) or not text:
            continue

        budget = _budget_for(field)
        if len(text) <= budget:
            continue

        attempt_text = text
        for attempt in range(1, settings.carousel_text_max_retries + 1):
            logger.warning(
                "carousel_field_exceeds_budget",
                field=field, length=len(attempt_text), budget=budget, attempt=attempt,
            )
            try:
                attempt_text = _regenerate_within_budget(field, attempt_text, budget, paper_title)
            except Exception as e:
                logger.warning("carousel_field_regen_failed", field=field, error=str(e))
                break
            if len(attempt_text) <= budget:
                break

        if len(attempt_text) > budget:
            logger.error(
                "carousel_field_budget_enforcement_failed",
                field=field, length=len(attempt_text), budget=budget,
            )
            attempt_text = _trim_to_boundary(attempt_text, budget)

        result[field] = attempt_text

    return result

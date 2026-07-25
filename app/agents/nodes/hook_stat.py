"""
Hook-stat extraction — pulls the single most surprising concrete number out
of a paper's results for a dedicated hero-number slide, instead of the
carousel opening with a paragraph. "One message per slide" applied to the
strongest message the deck has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.llm_retry import invoke_with_schema_retry
from app.agents.nodes.text_utils import normalize_model_strings
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class HookStat(BaseModel):
    value: str = Field(description="The single most surprising number, as a short string, e.g. '4', '10x', '89.2%'")
    label: str = Field(description="Short label under the number, max ~50 chars, e.g. 'denoising steps vs 50 in prior work'")


def extract_hook_stat_node(state: "PipelineState") -> dict:
    analysis = state.get("deep_analysis", {})
    breakthroughs = analysis.get("breakthroughs", "")
    results = "\n".join(analysis.get("quantitative_results", []))
    combined = f"{breakthroughs}\n\n{results}".strip()
    if not combined:
        return {"hook_stat": {}, "current_step": "hook_stat_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            api_key=settings.google_api_key,
        ).with_structured_output(HookStat)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Find the single most surprising, concrete number in this paper's results — "
             "the one number that alone would make an engineer stop scrolling. Prefer a "
             "small, striking number (a count, a multiplier, a percentage) over a big vague one. "
             "Return it standing alone, plus a short label explaining what it means."),
            ("user", "{text}"),
        ])

        stat: HookStat = invoke_with_schema_retry(
            lambda: (prompt | llm).invoke({"text": combined}),
            max_retries=settings.llm_schema_max_retries,
            logger=logger,
            context="hook_stat_extraction",
        )
        stat = normalize_model_strings(stat)  # type: ignore[assignment]
    except Exception as e:
        logger.warning("hook_stat_extraction_failed", error=str(e))
        return {"hook_stat": {}, "current_step": "hook_stat_skipped"}

    logger.info("hook_stat_extracted", value=stat.value, label=stat.label[:60])
    return {"hook_stat": stat.model_dump(), "current_step": "hook_stat_extracted"}

"""
Ablation component extraction — renders "which components were actually
validated by an ablation" as a row of chips instead of prose, with
components the paper never ablated shown visibly greyed-out rather than
silently omitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.llm_retry import invoke_with_schema_retry
from app.agents.nodes.text_utils import normalize_text
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AblationComponent(BaseModel):
    name: str = Field(description="Short component/module name, e.g. 'Sparse Routing'")
    validated: bool = Field(
        description="True if the paper reports an ablation removing this component; "
        "False if it's part of the architecture but never actually ablated/tested"
    )


class AblationSpecExtraction(BaseModel):
    components: list[AblationComponent] = Field(
        description="2-6 architectural components discussed in the methodology or ablation study"
    )


def _render_chips_html(components: list[AblationComponent]) -> str:
    chips = []
    for c in components:
        if c.validated:
            style = (
                "background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.35);"
                "color:#7C3AED;font-weight:700"
            )
            suffix = "✓ Validated"
        else:
            style = "background:#F1F5F9;border:1px dashed #CBD5E1;color:#94A3B8"
            suffix = "Not ablated"
        chips.append(
            f'<div style="{style};border-radius:8px;padding:12px 16px;text-align:center;min-width:120px">'
            f'<div style="font-size:14px;margin-bottom:4px">{c.name}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'text-transform:uppercase;letter-spacing:1px">{suffix}</div>'
            f"</div>"
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">'
        + "".join(chips)
        + "</div>"
    )


def extract_ablation_spec_node(state: "PipelineState") -> dict:
    analysis = state.get("deep_analysis", {})
    ablation_highlights = analysis.get("ablation_highlights", "")
    methodology = analysis.get("methodology", "")
    if not ablation_highlights and not methodology:
        return {"ablation_components": [], "ablation_chips_html": "", "current_step": "ablation_spec_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            api_key=settings.google_api_key,
        ).with_structured_output(AblationSpecExtraction)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "List the 2-6 main architectural components discussed in the methodology. "
             "Mark a component `validated: true` ONLY if the ablation text explicitly "
             "describes removing/disabling it and reporting the effect. Otherwise `false` — "
             "it's fine and expected for most papers to have unvalidated components."),
            ("user", "Methodology:\n{methodology}\n\nAblation study:\n{ablation}"),
        ])

        extraction: AblationSpecExtraction = invoke_with_schema_retry(
            lambda: (prompt | llm).invoke({
                "methodology": methodology,
                "ablation": ablation_highlights,
            }),
            max_retries=settings.llm_schema_max_retries,
            logger=logger,
            context="ablation_spec_extraction",
        )
    except Exception as e:
        logger.warning("ablation_spec_extraction_failed", error=str(e))
        return {"ablation_components": [], "ablation_chips_html": "", "current_step": "ablation_spec_skipped"}

    if not extraction.components:
        return {"ablation_components": [], "ablation_chips_html": "", "current_step": "ablation_spec_skipped"}

    components = [
        c.model_copy(update={"name": normalize_text(c.name)})
        for c in extraction.components[:6]
    ]
    chips_html = _render_chips_html(components)
    logger.info(
        "ablation_spec_extracted",
        total=len(components),
        validated=sum(1 for c in components if c.validated),
    )
    return {
        "ablation_components": [c.model_dump() for c in components],
        "ablation_chips_html": chips_html,
        "current_step": "ablation_spec_extracted",
    }

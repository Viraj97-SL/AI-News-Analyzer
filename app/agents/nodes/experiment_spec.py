"""
Structured experiment-setup extraction — turns the free-text
`experiment_setup` paragraph into a 4-field spec (Dataset / Baselines /
Metrics / Compute) for the carousel's spec-card grid. Missing fields are
returned as the literal "NOT REPORTED" so the template can amber-flag them
instead of the deck silently going quiet on gaps.
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

NOT_REPORTED = "NOT REPORTED"


class ExperimentSpec(BaseModel):
    dataset: str = Field(description=f"Dataset(s) used, or exactly '{NOT_REPORTED}' if not mentioned")
    baselines: str = Field(description=f"Baseline methods compared against, or exactly '{NOT_REPORTED}'")
    metrics: str = Field(description=f"Evaluation metrics used, or exactly '{NOT_REPORTED}'")
    compute: str = Field(description=f"Compute budget / hardware used, or exactly '{NOT_REPORTED}'")


def extract_experiment_spec_node(state: "PipelineState") -> dict:
    analysis = state.get("deep_analysis", {})
    experiment_setup = analysis.get("experiment_setup", "")
    if not experiment_setup:
        return {"experiment_spec": {}, "current_step": "experiment_spec_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            api_key=settings.google_api_key,
        ).with_structured_output(ExperimentSpec)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"Extract these 4 fields from the experiment setup text. If a field is "
             f"genuinely not mentioned in the text, return exactly '{NOT_REPORTED}' — "
             f"never guess or infer a plausible-sounding value."),
            ("user", "{experiment_setup}"),
        ])

        spec: ExperimentSpec = invoke_with_schema_retry(
            lambda: (prompt | llm).invoke({"experiment_setup": experiment_setup}),
            max_retries=settings.llm_schema_max_retries,
            logger=logger,
            context="experiment_spec_extraction",
        )
        spec = normalize_model_strings(spec)  # type: ignore[assignment]
    except Exception as e:
        logger.warning("experiment_spec_extraction_failed", error=str(e))
        return {"experiment_spec": {}, "current_step": "experiment_spec_skipped"}

    logger.info(
        "experiment_spec_extracted",
        not_reported_count=sum(1 for v in spec.model_dump().values() if v.strip().upper() == NOT_REPORTED),
    )
    return {"experiment_spec": spec.model_dump(), "current_step": "experiment_spec_extracted"}

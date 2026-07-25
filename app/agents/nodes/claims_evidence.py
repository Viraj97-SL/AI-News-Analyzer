"""
Claims-vs-evidence extraction for the Results slide.

Replaces a wall of prose ("the paper claims strong results...") with a
structured table: the claim on one side, an honest evidence-strength rating
on the other. This surfaces exactly where a paper is hand-waving instead of
letting prose smooth over the gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


class ClaimEvidence(BaseModel):
    claim: str = Field(description="A specific claim the paper makes, as a short phrase")
    evidence_status: Literal["quantified", "qualitative", "absent"] = Field(
        description=(
            "quantified = backed by a specific reported number; "
            "qualitative = described but with no hard number; "
            "absent = claimed but no evidence given at all"
        )
    )


class ClaimsEvidenceExtraction(BaseModel):
    claims: list[ClaimEvidence] = Field(description="3-6 claims the paper makes, each rated for evidence strength")


def extract_claims_evidence_node(state: "PipelineState") -> dict:
    analysis = state.get("deep_analysis", {})
    breakthroughs = analysis.get("breakthroughs", "")
    results_text = "\n".join(analysis.get("quantitative_results", []))
    combined = f"{breakthroughs}\n\n{results_text}".strip()
    if not combined:
        return {"claims_evidence": [], "current_step": "claims_evidence_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            api_key=settings.google_api_key,
        ).with_structured_output(ClaimsEvidenceExtraction)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract 3-6 distinct claims this paper makes about its own results, and rate "
             "each one's evidence strength. Be strict: a claim with a specific number is "
             "'quantified'; a claim described only in words ('substantially better') is "
             "'qualitative'; a claim with no supporting evidence at all is 'absent'."),
            ("user", "{text}"),
        ])

        extraction: ClaimsEvidenceExtraction = invoke_with_schema_retry(
            lambda: (prompt | llm).invoke({"text": combined}),
            max_retries=settings.llm_schema_max_retries,
            logger=logger,
            context="claims_evidence_extraction",
        )
    except Exception as e:
        logger.warning("claims_evidence_extraction_failed", error=str(e))
        return {"claims_evidence": [], "current_step": "claims_evidence_skipped"}

    claims = [
        c.model_copy(update={"claim": normalize_text(c.claim)})
        for c in extraction.claims[:6]
    ]
    logger.info("claims_evidence_extracted", count=len(claims))
    return {"claims_evidence": [c.model_dump() for c in claims], "current_step": "claims_evidence_extracted"}

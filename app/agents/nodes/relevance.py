"""
Paper relevance gate — scores the selected paper's fit for the target
audience before committing to a full carousel deep-dive.

Runs right after `deep_analysis_node` so scoring has real extracted content
to work with, not just the raw abstract. Papers below
`settings.relevance_score_threshold` skip the full visual pipeline (scoring,
benchmark chart, architecture diagram, prior art, carousel) and get a
lightweight digest entry instead — see `digest.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.llm_retry import invoke_with_schema_retry
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PaperRelevance(BaseModel):
    practitioner_actionability: int = Field(
        ge=1, le=10, description="Can a working engineer apply this in the next 6 months?"
    )
    audience_fit: int = Field(ge=1, le=10, description="How well does this match an AI/ML engineering audience?")
    claim_verifiability: int = Field(
        ge=1, le=10, description="Are real reported numbers present, or is this abstract-only hand-waving?"
    )
    novelty: int = Field(ge=1, le=10, description="How new is the core approach?")
    hype_penalty: int = Field(
        ge=0, le=10,
        description="Overclaiming, no code/data released, marketing language — HIGHER is worse.",
    )
    reasoning: str = Field(description="One sentence justifying the scores.")

    @property
    def score(self) -> float:
        return (
            self.practitioner_actionability
            + self.audience_fit
            + self.claim_verifiability
            + self.novelty
            - self.hype_penalty
        ) / 4.0


def relevance_gate_node(state: "PipelineState") -> dict:
    """Score the chosen paper's relevance; gate the full deep-dive pipeline on the result."""
    paper = state.get("chosen_research_paper", {})
    analysis = state.get("deep_analysis", {})
    if not paper or not analysis:
        return {"relevance_score": 0.0, "relevance_passed": False, "current_step": "relevance_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            api_key=settings.google_api_key,
        ).with_structured_output(PaperRelevance)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Score this AI research paper's fit for a working AI/ML engineering audience "
             "on 5 dimensions. Be critical and calibrated — most papers are NOT immediately "
             "actionable, are NOT highly novel, and DO overclaim relative to their evidence.\n"
             "practitioner_actionability: could an engineer apply this in the next 6 months?\n"
             "claim_verifiability: are there real reported numbers, or vague/abstract-only claims?\n"
             "hype_penalty (0-10, HIGHER is worse): overclaiming, no code/data released, marketing language."),
            ("user",
             "Title: {title}\n\nCore Problem: {core_problem}\n\nMethodology: {methodology}\n\n"
             "Breakthroughs: {breakthroughs}\n\nQuantitative Results:\n{results}"),
        ])

        relevance: PaperRelevance = invoke_with_schema_retry(
            lambda: (prompt | llm).invoke({
                "title": paper.get("title", ""),
                "core_problem": analysis.get("core_problem", ""),
                "methodology": analysis.get("methodology", ""),
                "breakthroughs": analysis.get("breakthroughs", ""),
                "results": "\n".join(analysis.get("quantitative_results", [])),
            }),
            max_retries=settings.llm_schema_max_retries,
            logger=logger,
            context="relevance_scoring",
        )
    except Exception as e:
        logger.warning("relevance_scoring_failed", error=str(e))
        # Fail open: an unscoreable paper still gets its deep-dive rather than silently vanishing.
        return {
            "relevance_score": settings.relevance_score_threshold,
            "relevance_passed": True,
            "current_step": "relevance_scoring_failed_default_pass",
        }

    passed = relevance.score >= settings.relevance_score_threshold
    logger.info(
        "paper_relevance_scored",
        title=paper.get("title", "")[:80],
        score=round(relevance.score, 2),
        threshold=settings.relevance_score_threshold,
        passed=passed,
        actionability=relevance.practitioner_actionability,
        audience_fit=relevance.audience_fit,
        claim_verifiability=relevance.claim_verifiability,
        novelty=relevance.novelty,
        hype_penalty=relevance.hype_penalty,
    )
    return {
        "relevance_score": round(relevance.score, 2),
        "relevance_passed": passed,
        "relevance_reasoning": relevance.reasoning,
        "current_step": "relevance_scored",
    }

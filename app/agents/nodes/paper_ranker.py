"""
Paper ranker node — scores all candidate papers on 5 dimensions before selection.

Produces a sorted paper_rankings list that select_paper_node uses instead of
asking the LLM to blindly pick one paper from a list. Manual papers receive a
strong composite-score boost so they always outrank scraped ArXiv papers.

Ranking dimensions (all 1-10):
  novelty          — how new / unique is the core approach?
  impact           — breadth and significance if the method works?
  technical_depth  — rigor, math, architecture detail?
  benchmark_quality — strong, diverse, fair evaluations?
  reproducibility  — code / data / weights released?

Composite formula:
  0.30 × novelty + 0.25 × impact + 0.20 × technical_depth
  + 0.15 × benchmark_quality + 0.10 × reproducibility
  [+ 3.0 flat bonus for manual/curated papers]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_BATCH_SIZE = 10  # papers per LLM call (keeps prompts within context limits)


class _PaperScore(BaseModel):
    paper_url: str = Field(description="Exact URL of the paper being scored.")
    novelty: int = Field(ge=1, le=10, description="Novelty 1-10: how new is the core approach?")
    impact: int = Field(ge=1, le=10, description="Impact 1-10: breadth and significance?")
    technical_depth: int = Field(
        ge=1, le=10, description="Technical depth 1-10: rigor, math, architecture detail?"
    )
    benchmark_quality: int = Field(
        ge=1, le=10, description="Benchmark quality 1-10: strong, diverse, fair evaluations?"
    )
    reproducibility: int = Field(
        ge=1, le=10, description="Reproducibility 1-10: code/data/weights released?"
    )
    reason: str = Field(description="One sentence on what makes this paper notable or weak.")


class _BatchRanking(BaseModel):
    rankings: list[_PaperScore] = Field(
        description="One ranking entry per paper in the batch — must include ALL papers."
    )


def rank_papers_node(state: "PipelineState") -> dict:
    """Score all raw_articles; return sorted paper_rankings."""
    articles = state.get("raw_articles", [])
    if not articles:
        return {"paper_rankings": [], "current_step": "ranking_skipped"}

    logger.info("ranking_papers", count=len(articles))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        api_key=settings.google_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a senior AI researcher. Score each paper on 5 dimensions (1-10 each).\n"
            "Calibration guide:\n"
            "  10 = paradigm shift (Transformer/BERT/AlphaFold level)\n"
            "   8 = strong novel contribution, likely cited widely\n"
            "   6 = solid work with clear improvements\n"
            "   4 = incremental, narrow, or heavily engineering-driven\n"
            "   2 = derivative, low novelty, weak evaluation\n"
            "Score ALL papers provided — omitting any paper is an error.",
        ),
        ("user", "{papers}"),
    ])

    all_scores: list[dict] = []

    for i in range(0, len(articles), _BATCH_SIZE):
        batch = articles[i : i + _BATCH_SIZE]
        papers_text = "\n\n---\n\n".join(
            f"URL: {a['url']}\nTitle: {a['title']}\nAbstract:\n{a['content'][:600]}"
            for a in batch
        )
        try:
            result: _BatchRanking = (
                prompt | llm.with_structured_output(_BatchRanking)
            ).invoke({"papers": papers_text})
            all_scores.extend(s.model_dump() for s in result.rankings)
            logger.info("ranking_batch_done", batch_start=i, scored=len(result.rankings))
        except Exception as e:
            logger.warning("ranking_batch_failed", batch_start=i, error=str(e))
            # Neutral defaults so the batch still participates in sorting
            for a in batch:
                all_scores.append({
                    "paper_url": a["url"],
                    "novelty": 5, "impact": 5, "technical_depth": 5,
                    "benchmark_quality": 5, "reproducibility": 5,
                    "reason": "Ranking unavailable — defaults applied.",
                })

    # Identify manually curated papers for priority boost
    manual_urls = {a["url"] for a in articles if a.get("source") == "manual"}

    for s in all_scores:
        composite = (
            s.get("novelty", 5) * 0.30
            + s.get("impact", 5) * 0.25
            + s.get("technical_depth", 5) * 0.20
            + s.get("benchmark_quality", 5) * 0.15
            + s.get("reproducibility", 5) * 0.10
        )
        if s.get("paper_url") in manual_urls:
            composite += 3.0  # ensures manual papers always win selection
        s["composite_score"] = round(composite, 2)
        s["is_manual"] = s.get("paper_url") in manual_urls

    all_scores.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    if all_scores:
        top = all_scores[0]
        logger.info(
            "papers_ranked",
            total=len(all_scores),
            top_score=top.get("composite_score"),
            top_url=top.get("paper_url", "")[:60],
            is_manual=top.get("is_manual", False),
        )

    return {"paper_rankings": all_scores, "current_step": "papers_ranked"}

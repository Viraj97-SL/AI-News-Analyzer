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

Composite formula (research pipeline v2):
  quality = 0.30 x novelty + 0.25 x impact + 0.20 x technical_depth
          + 0.15 x benchmark_quality + 0.10 x reproducibility
  composite = quality
            + lab_bonus        (superlab +1.5, tier-2 lab +0.5 — decision D4)
            + attention_bonus  (HF upvotes / GitHub stars, log-scaled, max +2.0)
            + trend_bonus      (+0.5 per hot trend, max +1.0 — decision D6)
            [+ 3.0 for manual/curated papers, which also always sort first]

Candidates from all research sources are first deduped/enriched
(`merge_research_candidates`), then capped to MAX_LLM_RANKED by a cheap
signal-only pre-score so LLM ranking cost stays bounded as sources grow.\n"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.agents.nodes.lab_registry import LAB_TIER_BONUS
from app.agents.nodes.research_sources import attention_bonus, merge_research_candidates
from app.agents.nodes.trends import trend_bonus
from app.core.config import get_settings
from app.core.models import research_model
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_BATCH_SIZE = 10  # papers per LLM call (keeps prompts within context limits)
MAX_LLM_RANKED = 40  # cap on papers sent to the LLM ranker per run
MANUAL_BONUS = 3.0
_QUALITY_WEIGHTS = {
    "novelty": 0.30,
    "impact": 0.25,
    "technical_depth": 0.20,
    "benchmark_quality": 0.15,
    "reproducibility": 0.10,
}
_NEUTRAL_SCORE = 5


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
    is_ai_ml: bool = Field(
        default=True,
        description=(
            "True if the paper's main subject is AI/ML: models, training, inference, agents, "
            "evaluation, data for ML, or ML applied to a domain. False for pure math, "
            "algorithms/complexity, physics, biology, etc. with no ML focus."
        ),
    )


class _BatchRanking(BaseModel):
    rankings: list[_PaperScore] = Field(
        description="One ranking entry per paper in the batch — must include ALL papers."
    )


def _lab_bonus_for(article: dict) -> float:
    return LAB_TIER_BONUS.get(article.get("lab_tier") or 0, 0.0)


def _signal_prescore(article: dict) -> float:
    return (
        _lab_bonus_for(article)
        + attention_bonus(article)
        + trend_bonus(article.get("trend_tags") or [])
    )


def select_for_llm_ranking(candidates: list[dict]) -> list[dict]:
    """All manual papers + the highest signal pre-score papers, up to MAX_LLM_RANKED."""
    manual = [a for a in candidates if a.get("source") == "manual"]
    others = [a for a in candidates if a.get("source") != "manual"]
    # sorted() is stable, so equal pre-scores keep source order (newest arXiv first)
    ranked_others = sorted(others, key=_signal_prescore, reverse=True)
    return (manual + ranked_others)[: max(MAX_LLM_RANKED, len(manual))]


def _build_ranking_chain():
    llm = ChatGoogleGenerativeAI(
        model=research_model("fast"),
        temperature=0.1,
        api_key=settings.google_api_key,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior AI researcher. Score each paper on 5 dimensions (1-10 each).\n"
                "Calibration guide:\n"
                "  10 = paradigm shift (Transformer/BERT/AlphaFold level)\n"
                "   8 = strong novel contribution, likely cited widely\n"
                "   6 = solid work with clear improvements\n"
                "   4 = incremental, narrow, or heavily engineering-driven\n"
                "   2 = derivative, low novelty, weak evaluation\n"
                "Also set is_ai_ml: this digest is for an AI/ML audience, so mark papers whose "
                "main subject is not AI/ML (pure math, complexity theory, physics...) false.\n"
                "Score ALL papers provided — omitting any paper is an error.",
            ),
            ("user", "{papers}"),
        ]
    )
    return prompt | llm.with_structured_output(_BatchRanking)


def _llm_quality_scores(candidates: list[dict]) -> dict[str, dict]:
    """url -> raw LLM dimension scores; failed batches get neutral defaults."""
    chain = _build_ranking_chain()
    scores: dict[str, dict] = {}
    for i in range(0, len(candidates), _BATCH_SIZE):
        batch = candidates[i : i + _BATCH_SIZE]
        papers_text = "\n\n---\n\n".join(
            f"URL: {a['url']}\nTitle: {a['title']}\nAbstract:\n{a['content'][:600]}" for a in batch
        )
        try:
            result: _BatchRanking = chain.invoke({"papers": papers_text})
            for s in result.rankings:
                scores[s.paper_url] = s.model_dump()
            logger.info("ranking_batch_done", batch_start=i, scored=len(result.rankings))
        except Exception as e:
            logger.warning("ranking_batch_failed", batch_start=i, error=str(e))
    return scores


def _drop_off_topic(candidates: list[dict], llm_scores: dict[str, dict]) -> list[dict]:
    """Remove papers the ranker flagged as not AI/ML. User-queued (manual) papers and
    papers without a score (failed batch) are kept: the filter fails open."""
    kept = [
        a for a in candidates
        if a.get("source") == "manual" or llm_scores.get(a["url"], {}).get("is_ai_ml", True)
    ]
    dropped = [a["title"][:80] for a in candidates if a not in kept]
    if dropped:
        logger.info("off_topic_papers_dropped", count=len(dropped), titles=dropped)
    return kept


def _score_entry(article: dict, llm_scores: dict | None) -> dict:
    dims = {k: (llm_scores or {}).get(k, _NEUTRAL_SCORE) for k in _QUALITY_WEIGHTS}
    quality = sum(dims[k] * w for k, w in _QUALITY_WEIGHTS.items())
    is_manual = article.get("source") == "manual"
    lab_b = _lab_bonus_for(article)
    attn_b = attention_bonus(article)
    trend_b = trend_bonus(article.get("trend_tags") or [])
    composite = quality + lab_b + attn_b + trend_b + (MANUAL_BONUS if is_manual else 0.0)
    return {
        "paper_url": article["url"],
        **dims,
        "reason": (llm_scores or {}).get("reason", "Ranking unavailable — defaults applied."),
        "quality_score": round(quality, 2),
        "lab": article.get("lab"),
        "lab_bonus": lab_b,
        "attention_bonus": attn_b,
        "trend_tags": list(article.get("trend_tags") or []),
        "trend_bonus": trend_b,
        "is_manual": is_manual,
        "composite_score": round(composite, 2),
    }


def rank_papers_node(state: PipelineState) -> dict:
    """Merge candidates across sources, score them, return sorted paper_rankings."""
    articles = state.get("raw_articles", [])
    if not articles:
        return {"paper_rankings": [], "research_candidates": [], "current_step": "ranking_skipped"}

    candidates = merge_research_candidates(articles)
    to_rank = select_for_llm_ranking(candidates)
    logger.info(
        "ranking_papers", raw=len(articles), unique=len(candidates), llm_ranked=len(to_rank)
    )

    llm_scores = _llm_quality_scores(to_rank)
    to_rank = _drop_off_topic(to_rank, llm_scores)
    rankings = sorted(
        (_score_entry(a, llm_scores.get(a["url"])) for a in to_rank),
        # Manual (user-curated) papers always win regardless of boosts; then composite.
        key=lambda r: (r["is_manual"], r["composite_score"]),
        reverse=True,
    )

    if rankings:
        top = rankings[0]
        logger.info(
            "papers_ranked",
            total=len(rankings),
            top_score=top["composite_score"],
            top_url=top["paper_url"][:60],
            top_lab=top["lab"],
            top_trends=top["trend_tags"],
            is_manual=top["is_manual"],
        )

    return {
        "paper_rankings": rankings,
        "research_candidates": candidates,
        "current_step": "papers_ranked",
    }

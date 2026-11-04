"""
Per-run quality metrics for the research pipeline.

One flat JSON record per run is appended to a JSONL file. The file is the input
for ``app.core.drift`` (offline drift report) and the same values are attached
to the LangSmith trace as feedback scores when tracing is enabled.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS_PATH = Path("./output/metrics/research_runs.jsonl")

VERDICT_RANK: dict[str, int] = {
    "Incremental": 1,
    "Solid Contribution": 2,
    "Major Contribution": 3,
    "Paradigm Shift": 4,
}


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def extract_research_metrics(state: dict[str, Any], usage: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten the final graph state + usage snapshot into one metrics record."""
    paper = state.get("chosen_research_paper") or {}
    analysis = state.get("deep_analysis") or {}
    scores = state.get("research_scores") or {}
    hook = state.get("hook_score") or {}
    usage = usage or {}

    verdict = analysis.get("significance_verdict")
    hook_values = [float(hook[k]) for k in ("curiosity", "specificity", "controversy") if k in hook]
    fill_ratios = [float(r) for r in state.get("slide_fill_ratios") or []]

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": state.get("run_id"),
        "trigger_type": state.get("trigger_type"),
        "paper_url": paper.get("url"),
        "paper_title": paper.get("title"),
        "paper_source": paper.get("source"),
        "lab": paper.get("lab"),
        "trend_tags": list(paper.get("trend_tags") or []),
        "is_classic": bool(state.get("is_classic_paper", False)),
        "relevance_score": state.get("relevance_score"),
        "relevance_passed": state.get("relevance_passed"),
        "verdict": verdict,
        "verdict_rank": VERDICT_RANK.get(verdict) if verdict else None,
        "novelty": scores.get("novelty"),
        "methodology_clarity": scores.get("methodology_clarity"),
        "benchmark_improvement": scores.get("benchmark_improvement"),
        "reproducibility": scores.get("reproducibility"),
        "hook_score_avg": _mean(hook_values),
        "full_text_available": state.get("full_text_available"),
        "figure_count": len(state.get("paper_figures") or []),
        "has_benchmark_chart": bool(state.get("benchmark_chart_path")),
        "slide_count": len(state.get("research_carousel_slide_paths") or []),
        "slide_fill_ratio_avg": _mean(fill_ratios),
        "error_count": len(state.get("error_log") or []),
        "approval_status": state.get("approval_status"),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "total_cost": float(usage.get("total_cost", 0.0)),
        "llm_calls": int(usage.get("llm_calls", 0)),
    }


def append_run_metrics(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def load_run_metrics(path: Path) -> list[dict[str, Any]]:
    """Load all records in file order; corrupt lines are skipped with a warning."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("run_metrics_corrupt_line", path=str(path), line=line_no)
    return records

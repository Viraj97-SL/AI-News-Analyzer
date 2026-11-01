"""
Offline drift detection over research run metrics (see ``run_metrics.py``).

Compares the mean of each tracked metric in the most recent ``recent_window``
runs against the preceding ``baseline_window`` runs. A metric drifts when the
absolute change exceeds its threshold. Booleans are treated as rates (0/1).

This is deliberately simple: the pipeline runs about weekly, so a handful of
windowed means catches the failure modes we care about (LLM score inflation,
verdict creep, falling full-text/figure extraction rates, cost creep).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# metric name → absolute change that counts as drift
DRIFT_THRESHOLDS: dict[str, float] = {
    "relevance_score": 1.5,  # 0-10 scale
    "verdict_rank": 0.75,  # 1-4 scale; creep toward "Paradigm Shift"
    "novelty": 1.5,
    "hook_score_avg": 1.5,
    "full_text_available": 0.34,  # rate
    "has_benchmark_chart": 0.34,
    "figure_count": 1.0,
    "slide_fill_ratio_avg": 0.15,
    "error_count": 2.0,
    "total_cost": 0.25,  # USD
}


@dataclass(frozen=True)
class DriftAlert:
    metric: str
    baseline: float
    recent: float
    threshold: float
    direction: str  # "up" | "down"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _window_mean(records: list[dict[str, Any]], metric: str) -> float | None:
    values = [v for v in (_numeric(r.get(metric)) for r in records) if v is not None]
    return sum(values) / len(values) if values else None


def detect_drift(
    records: list[dict[str, Any]],
    baseline_window: int = 8,
    recent_window: int = 4,
    thresholds: dict[str, float] | None = None,
) -> list[DriftAlert]:
    """Return one alert per drifting metric; empty when history is too short."""
    if len(records) < baseline_window + recent_window:
        return []
    recent = records[-recent_window:]
    baseline = records[-(baseline_window + recent_window) : -recent_window]

    alerts: list[DriftAlert] = []
    for metric, threshold in (thresholds or DRIFT_THRESHOLDS).items():
        base_mean = _window_mean(baseline, metric)
        recent_mean = _window_mean(recent, metric)
        if base_mean is None or recent_mean is None:
            continue
        delta = recent_mean - base_mean
        if abs(delta) > threshold:
            alerts.append(
                DriftAlert(
                    metric=metric,
                    baseline=round(base_mean, 4),
                    recent=round(recent_mean, 4),
                    threshold=threshold,
                    direction="up" if delta > 0 else "down",
                )
            )
    return alerts


def format_report(alerts: list[DriftAlert], n_records: int) -> str:
    lines = [f"Research pipeline drift report — {n_records} runs analysed"]
    if not alerts:
        lines.append("No drift detected.")
        return "\n".join(lines)
    for a in alerts:
        arrow = "▲" if a.direction == "up" else "▼"
        lines.append(f"  {arrow} {a.metric}: {a.baseline} → {a.recent} (threshold ±{a.threshold})")
    return "\n".join(lines)

"""Tests for research run metrics extraction/persistence and drift detection."""

from __future__ import annotations

import json

import pytest

from app.core.drift import DriftAlert, detect_drift, format_report
from app.core.run_metrics import (
    VERDICT_RANK,
    append_run_metrics,
    extract_research_metrics,
    load_run_metrics,
)


def _state(**overrides):
    base = {
        "run_id": "run-1",
        "trigger_type": "scheduled",
        "chosen_research_paper": {
            "url": "https://arxiv.org/abs/2601.00001",
            "title": "Paper",
            "source": "hf_daily",
            "lab": "Google DeepMind",
            "trend_tags": ["rag", "multi_agent"],
        },
        "deep_analysis": {"significance_verdict": "Major Contribution"},
        "relevance_score": 7.2,
        "relevance_passed": True,
        "research_scores": {
            "novelty": 8,
            "methodology_clarity": 7,
            "benchmark_improvement": 6,
            "reproducibility": 5,
        },
        "hook_score": {"curiosity": 8, "specificity": 6, "controversy": 4},
        "full_text_available": True,
        "paper_figures": [{"b64": "x"}, {"b64": "y"}],
        "benchmark_chart_path": "out/chart.png",
        "research_carousel_slide_paths": ["a.png", "b.png", "c.png"],
        "slide_fill_ratios": [0.8, 0.6],
        "error_log": ["e1"],
        "approval_status": "pending",
    }
    return {**base, **overrides}


USAGE = {"total_tokens": 1234, "total_cost": 0.05, "llm_calls": 9, "by_model": {}}


def test_extract_research_metrics_core_fields():
    m = extract_research_metrics(_state(), USAGE)
    assert m["run_id"] == "run-1"
    assert m["paper_url"] == "https://arxiv.org/abs/2601.00001"
    assert m["lab"] == "Google DeepMind"
    assert m["trend_tags"] == ["rag", "multi_agent"]
    assert m["verdict"] == "Major Contribution"
    assert m["verdict_rank"] == VERDICT_RANK["Major Contribution"]
    assert m["relevance_score"] == 7.2
    assert m["novelty"] == 8
    assert m["hook_score_avg"] == pytest.approx(6.0)
    assert m["full_text_available"] is True
    assert m["figure_count"] == 2
    assert m["has_benchmark_chart"] is True
    assert m["slide_count"] == 3
    assert m["slide_fill_ratio_avg"] == pytest.approx(0.7)
    assert m["error_count"] == 1
    assert m["total_tokens"] == 1234
    assert m["total_cost"] == 0.05
    assert "timestamp" in m


def test_extract_research_metrics_handles_sparse_digest_state():
    m = extract_research_metrics({"run_id": "r2", "relevance_passed": False}, None)
    assert m["run_id"] == "r2"
    assert m["verdict"] is None
    assert m["verdict_rank"] is None
    assert m["hook_score_avg"] is None
    assert m["slide_fill_ratio_avg"] is None
    assert m["figure_count"] == 0
    assert m["total_cost"] == 0.0


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "metrics" / "runs.jsonl"
    append_run_metrics(path, {"run_id": "a", "x": 1})
    append_run_metrics(path, {"run_id": "b", "x": 2})
    records = load_run_metrics(path)
    assert [r["run_id"] for r in records] == ["a", "b"]


def test_load_skips_corrupt_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"run_id": "a"}) + "\n{not json\n\n" + json.dumps({"run_id": "b"}) + "\n"
    )
    assert [r["run_id"] for r in load_run_metrics(path)] == ["a", "b"]


def test_load_missing_file_returns_empty(tmp_path):
    assert load_run_metrics(tmp_path / "nope.jsonl") == []


# ── drift ────────────────────────────────────────────────────────────────────


def _runs(n, **fields):
    return [{"run_id": f"r{i}", **fields} for i in range(n)]


def test_detect_drift_flags_relevance_score_inflation():
    records = _runs(6, relevance_score=5.0) + _runs(3, relevance_score=8.5)
    alerts = detect_drift(records, baseline_window=6, recent_window=3)
    names = {a.metric for a in alerts}
    assert "relevance_score" in names
    alert = next(a for a in alerts if a.metric == "relevance_score")
    assert alert.direction == "up"


def test_detect_drift_flags_falling_full_text_rate():
    records = _runs(6, full_text_available=True) + _runs(3, full_text_available=False)
    alerts = detect_drift(records, baseline_window=6, recent_window=3)
    alert = next(a for a in alerts if a.metric == "full_text_available")
    assert alert.direction == "down"


def test_detect_drift_quiet_when_stable():
    records = _runs(
        9, relevance_score=6.0, verdict_rank=2, full_text_available=True, total_cost=0.1
    )
    assert detect_drift(records, baseline_window=6, recent_window=3) == []


def test_detect_drift_needs_enough_history():
    assert detect_drift(_runs(2, relevance_score=1.0), baseline_window=6, recent_window=3) == []


def test_detect_drift_ignores_missing_values():
    records = _runs(6, relevance_score=6.0) + [{"run_id": "x"}] * 3
    assert all(a.metric != "relevance_score" for a in detect_drift(records, 6, 3))


def test_format_report_lists_alerts_and_ok():
    alert = DriftAlert(
        metric="total_cost", baseline=0.1, recent=0.5, threshold=0.25, direction="up"
    )
    assert "total_cost" in format_report([alert], n_records=10)
    assert "No drift" in format_report([], n_records=10)

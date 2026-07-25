"""Tests for verdict.py — deterministic significance-verdict calibration."""

from __future__ import annotations

from app.agents.nodes.verdict import calibrate_verdict


class TestCalibrateVerdict:
    def test_paradigm_shift_requires_all_three_thresholds(self):
        scores = {"benchmark_improvement": 9, "reproducibility": 8, "novelty": 9}
        assert calibrate_verdict(scores) == "Paradigm Shift"

    def test_high_benchmark_alone_is_not_paradigm_shift(self):
        """Reproducibility 1/10 must never coexist with the top verdict — the exact
        contradiction the reference deck exhibited."""
        scores = {"benchmark_improvement": 9, "reproducibility": 1, "novelty": 9}
        assert calibrate_verdict(scores) != "Paradigm Shift"

    def test_major_contribution_requires_benchmark_and_repro(self):
        scores = {"benchmark_improvement": 7, "reproducibility": 5, "novelty": 3}
        assert calibrate_verdict(scores) == "Major Contribution"

    def test_major_contribution_denied_without_reproducibility(self):
        """The reference-deck bug: MAJOR CONTRIBUTION next to a 1/10 repro gauge."""
        scores = {"benchmark_improvement": 9, "reproducibility": 1, "novelty": 3}
        assert calibrate_verdict(scores) != "Major Contribution"
        assert calibrate_verdict(scores) != "Paradigm Shift"

    def test_solid_contribution_on_moderate_benchmark(self):
        scores = {"benchmark_improvement": 5, "reproducibility": 1, "novelty": 2}
        assert calibrate_verdict(scores) == "Solid Contribution"

    def test_solid_contribution_on_high_novelty_alone(self):
        scores = {"benchmark_improvement": 1, "reproducibility": 1, "novelty": 7}
        assert calibrate_verdict(scores) == "Solid Contribution"

    def test_incremental_for_low_scores_across_the_board(self):
        scores = {"benchmark_improvement": 2, "reproducibility": 1, "novelty": 2}
        assert calibrate_verdict(scores) == "Incremental"

    def test_missing_keys_default_to_zero_and_yield_incremental(self):
        assert calibrate_verdict({}) == "Incremental"

    def test_verdict_is_deterministic_pure_function(self):
        scores = {"benchmark_improvement": 7, "reproducibility": 6, "novelty": 5}
        assert calibrate_verdict(scores) == calibrate_verdict(scores)

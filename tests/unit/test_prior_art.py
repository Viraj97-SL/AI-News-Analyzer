"""Tests for prior_art.py's win-capping — 'one message per slide' fix for the
comparison table (previously every row could read WINS, i.e. zero contrast)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.prior_art import (
    ComparisonDimension,
    PriorArtComparison,
    _normalize_comparison,
    prior_art_node,
)


def _comparison(winners: list[str]) -> PriorArtComparison:
    return PriorArtComparison(
        prior_paper_name="Prior Method",
        overall_verdict="Net improvement.",
        dimensions=[
            ComparisonDimension(dimension=f"Dim {i}", new_paper="better", prior_sota="worse", winner=w)
            for i, w in enumerate(winners)
        ],
    )


class TestCapNewWins:
    def test_all_five_wins_capped_to_configured_max(self, monkeypatch):
        from app.agents.nodes import prior_art as prior_art_module

        monkeypatch.setattr(prior_art_module.settings, "prior_art_max_wins", 2)
        comparison = _comparison(["new", "new", "new", "new", "new"])

        result = _normalize_comparison(comparison)

        win_count = sum(1 for d in result.dimensions if d.winner == "new")
        assert win_count == 2

    def test_first_n_wins_preserved_rest_demoted_to_tie(self, monkeypatch):
        from app.agents.nodes import prior_art as prior_art_module

        monkeypatch.setattr(prior_art_module.settings, "prior_art_max_wins", 2)
        comparison = _comparison(["new", "new", "new"])

        result = _normalize_comparison(comparison)

        assert result.dimensions[0].winner == "new"
        assert result.dimensions[1].winner == "new"
        assert result.dimensions[2].winner == "tie"

    def test_fewer_wins_than_cap_untouched(self, monkeypatch):
        from app.agents.nodes import prior_art as prior_art_module

        monkeypatch.setattr(prior_art_module.settings, "prior_art_max_wins", 2)
        comparison = _comparison(["new", "prior", "tie"])

        result = _normalize_comparison(comparison)

        assert result.dimensions[0].winner == "new"
        assert result.dimensions[1].winner == "prior"
        assert result.dimensions[2].winner == "tie"

    def test_prior_wins_never_capped(self, monkeypatch):
        from app.agents.nodes import prior_art as prior_art_module

        monkeypatch.setattr(prior_art_module.settings, "prior_art_max_wins", 1)
        comparison = _comparison(["prior", "prior", "prior"])

        result = _normalize_comparison(comparison)

        assert all(d.winner == "prior" for d in result.dimensions)

    def test_normalize_still_strips_latex_after_capping(self, monkeypatch):
        from app.agents.nodes import prior_art as prior_art_module

        monkeypatch.setattr(prior_art_module.settings, "prior_art_max_wins", 5)
        comparison = PriorArtComparison(
            prior_paper_name="Prior $O(n^2)$ Method",
            overall_verdict="Better.",
            dimensions=[
                ComparisonDimension(dimension="Speed", new_paper="$10^4$ ops/s", prior_sota="slow", winner="new"),
            ],
        )
        result = _normalize_comparison(comparison)
        assert "$" not in result.prior_paper_name
        assert "$" not in result.dimensions[0].new_paper


class TestPriorArtNodeRetry:
    def _comparison_result(self) -> PriorArtComparison:
        return PriorArtComparison(
            prior_paper_name="Prior Method",
            overall_verdict="Net improvement.",
            dimensions=[
                ComparisonDimension(
                    dimension="Accuracy", new_paper="92%", prior_sota="88%", winner="new",
                ),
            ],
        )

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        from app.agents.nodes import prior_art as prior_art_module

        monkeypatch.setattr(prior_art_module.settings, "llm_schema_max_retries", 2)
        comparison = self._comparison_result()
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), comparison]

        with patch("app.agents.nodes.prior_art.ChatGoogleGenerativeAI") as MockLLM, \
                patch("app.agents.nodes.screenshot_utils.capture_slide", return_value="output/images/card.png"), \
                patch("app.agents.nodes.screenshot_utils.make_hti", return_value=MagicMock()):
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = prior_art_node({
                "chosen_research_paper": {"title": "Some Paper"},
                "deep_analysis": {
                    "core_problem": "A problem.",
                    "methodology": "A method.",
                    "breakthroughs": "Results.",
                },
                "run_id": "test-run",
            })

        assert result["current_step"] == "prior_art_card_generated"
        assert result["prior_art_comparison"]["prior_paper_name"] == "Prior Method"
        assert mock_llm.call_count == 2

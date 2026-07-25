"""Tests for relevance.py — the pre-carousel relevance gate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.relevance import PaperRelevance, relevance_gate_node


def _mock_llm_instance(relevance: PaperRelevance) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.return_value = relevance          # callable path
    mock_llm.invoke.return_value = relevance    # .invoke() path
    return mock_llm


def _base_state(**overrides) -> dict:
    base = {
        "chosen_research_paper": {"title": "Some Paper", "url": "https://arxiv.org/abs/1"},
        "deep_analysis": {
            "core_problem": "A problem.",
            "methodology": "A method.",
            "breakthroughs": "Results.",
            "quantitative_results": ["MMLU: 90%"],
        },
    }
    base.update(overrides)
    return base


class TestPaperRelevanceScore:
    def test_score_is_mean_of_four_dims_minus_hype(self):
        r = PaperRelevance(
            practitioner_actionability=8, audience_fit=8, claim_verifiability=8, novelty=8,
            hype_penalty=0, reasoning="x",
        )
        assert r.score == 8.0

    def test_hype_penalty_reduces_score(self):
        r = PaperRelevance(
            practitioner_actionability=8, audience_fit=8, claim_verifiability=8, novelty=8,
            hype_penalty=8, reasoning="x",
        )
        assert r.score == 6.0


class TestRelevanceGateNode:
    def test_skips_when_no_paper(self):
        result = relevance_gate_node({"chosen_research_paper": {}, "deep_analysis": {}})
        assert result["relevance_passed"] is False
        assert result["current_step"] == "relevance_skipped"

    def test_passes_when_score_above_threshold(self, monkeypatch):
        from app.agents.nodes import relevance as relevance_module

        monkeypatch.setattr(relevance_module.settings, "relevance_score_threshold", 5.0)
        relevance = PaperRelevance(
            practitioner_actionability=8, audience_fit=8, claim_verifiability=8, novelty=8,
            hype_penalty=0, reasoning="Strong actionable paper.",
        )
        with patch("app.agents.nodes.relevance.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(relevance)
            MockLLM.return_value = mock_instance

            result = relevance_gate_node(_base_state())

        assert result["relevance_passed"] is True
        assert result["current_step"] == "relevance_scored"

    def test_fails_when_score_below_threshold(self, monkeypatch):
        from app.agents.nodes import relevance as relevance_module

        monkeypatch.setattr(relevance_module.settings, "relevance_score_threshold", 5.0)
        relevance = PaperRelevance(
            practitioner_actionability=2, audience_fit=2, claim_verifiability=2, novelty=2,
            hype_penalty=5, reasoning="Weak, hype-heavy paper.",
        )
        with patch("app.agents.nodes.relevance.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(relevance)
            MockLLM.return_value = mock_instance

            result = relevance_gate_node(_base_state())

        assert result["relevance_passed"] is False

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        """A schema/validation failure on the first attempt should retry rather than
        immediately falling back to the fail-open default."""
        from app.agents.nodes import relevance as relevance_module

        monkeypatch.setattr(relevance_module.settings, "relevance_score_threshold", 5.0)
        monkeypatch.setattr(relevance_module.settings, "llm_schema_max_retries", 2)
        relevance = PaperRelevance(
            practitioner_actionability=9, audience_fit=9, claim_verifiability=9, novelty=9,
            hype_penalty=0, reasoning="Recovered after retry.",
        )
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), relevance]
        with patch("app.agents.nodes.relevance.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = relevance_gate_node(_base_state())

        assert result["current_step"] == "relevance_scored"
        assert result["relevance_passed"] is True
        assert mock_llm.call_count == 2

    def test_llm_failure_fails_open(self):
        """A paper that can't be scored should still get its deep-dive, not vanish."""
        with patch("app.agents.nodes.relevance.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = relevance_gate_node(_base_state())

        assert result["relevance_passed"] is True
        assert result["current_step"] == "relevance_scoring_failed_default_pass"

    def test_logs_every_score(self, monkeypatch, caplog):
        from app.agents.nodes import relevance as relevance_module

        monkeypatch.setattr(relevance_module.settings, "relevance_score_threshold", 5.0)
        relevance = PaperRelevance(
            practitioner_actionability=5, audience_fit=5, claim_verifiability=5, novelty=5,
            hype_penalty=0, reasoning="Average.",
        )
        with patch("app.agents.nodes.relevance.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(relevance)
            MockLLM.return_value = mock_instance

            result = relevance_gate_node(_base_state())

        assert "relevance_score" in result
        assert "relevance_reasoning" in result

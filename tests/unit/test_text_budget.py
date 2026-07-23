"""
Tests for text_budget.py — fitting carousel text fields to their slide's
character budget via LLM rewrite (with retries) before falling back to a
sentence/word-boundary trim.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes import text_budget
from app.agents.nodes.text_budget import enforce_char_budgets


def _mock_llm_instance(reply_content: str) -> MagicMock:
    """
    LangChain wraps a non-Runnable mock via RunnableLambda, so (prompt | mock).invoke()
    calls mock(input) — not mock.invoke(input). Set both so the test is robust to
    either call convention (same pattern as test_research_pipeline.py).
    """
    mock_response = MagicMock()
    mock_response.content = reply_content
    mock_llm = MagicMock()
    mock_llm.return_value = mock_response
    mock_llm.invoke.return_value = mock_response
    return mock_llm


class TestEnforceCharBudgets:
    def test_field_within_budget_untouched_and_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 500)
        analysis = {"core_problem": "Short text."}

        with patch("app.agents.nodes.text_budget.ChatGoogleGenerativeAI") as MockLLM:
            result = enforce_char_budgets(analysis, paper_title="Some Paper")

        assert result["core_problem"] == "Short text."
        MockLLM.assert_not_called()

    def test_field_over_budget_regenerated_via_llm(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 20)
        monkeypatch.setattr(text_budget.settings, "carousel_text_max_retries", 2)
        analysis = {"core_problem": "This is a very long sentence that exceeds the tiny budget."}

        with patch("app.agents.nodes.text_budget.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = _mock_llm_instance("Fits now.")
            result = enforce_char_budgets(analysis, paper_title="Some Paper")

        assert result["core_problem"] == "Fits now."

    def test_persistent_overage_falls_back_to_sentence_trim_not_ellipsis(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 20)
        monkeypatch.setattr(text_budget.settings, "carousel_text_max_retries", 2)
        long_text = "This sentence is far too long to ever fit. It keeps going regardless."
        analysis = {"core_problem": long_text}

        with patch("app.agents.nodes.text_budget.ChatGoogleGenerativeAI") as MockLLM:
            # LLM keeps returning text that's still too long — forces fallback trim.
            MockLLM.return_value = _mock_llm_instance(long_text)
            result = enforce_char_budgets(analysis, paper_title="Some Paper")

        trimmed = result["core_problem"]
        assert len(trimmed) <= 20
        assert not trimmed.endswith("...")

    def test_llm_failure_falls_back_to_trim_without_raising(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 15)
        monkeypatch.setattr(text_budget.settings, "carousel_text_max_retries", 2)
        analysis = {"core_problem": "This text will fail to regenerate via the LLM call entirely."}

        with patch("app.agents.nodes.text_budget.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = enforce_char_budgets(analysis, paper_title="Some Paper")

        assert len(result["core_problem"]) <= 15

    def test_fields_outside_budget_map_are_untouched(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 10)
        analysis = {"key_contributions": ["A very long contribution sentence that is not budget-managed."]}

        with patch("app.agents.nodes.text_budget.ChatGoogleGenerativeAI") as MockLLM:
            result = enforce_char_budgets(analysis, paper_title="Some Paper")

        assert result["key_contributions"] == analysis["key_contributions"]
        MockLLM.assert_not_called()

    def test_short_field_uses_short_budget(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_short_char_budget", 10)
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 500)
        analysis = {"limitations": "This limitation text is longer than ten characters."}

        with patch("app.agents.nodes.text_budget.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("skip regen, force fallback trim")
            result = enforce_char_budgets(analysis, paper_title="Some Paper")

        assert len(result["limitations"]) <= 10

    def test_does_not_mutate_input_dict(self, monkeypatch):
        monkeypatch.setattr(text_budget.settings, "carousel_body_char_budget", 500)
        analysis = {"core_problem": "Short."}
        result = enforce_char_budgets(analysis, paper_title="Some Paper")
        assert result is not analysis


class TestTrimToBoundary:
    def test_never_cuts_mid_word(self):
        text = "The quick brown fox jumps over the lazy dog and keeps running."
        trimmed = text_budget._trim_to_boundary(text, budget=20)
        assert len(trimmed) <= 20
        assert trimmed == "" or text.startswith(trimmed)

    def test_never_appends_ellipsis(self):
        text = "A" * 100
        trimmed = text_budget._trim_to_boundary(text, budget=10)
        assert "..." not in trimmed

    def test_short_text_passthrough(self):
        text = "Short."
        assert text_budget._trim_to_boundary(text, budget=100) == text

    def test_prefers_sentence_boundary_when_available(self):
        text = "First sentence here. Second sentence that would overflow the budget by a lot."
        trimmed = text_budget._trim_to_boundary(text, budget=25)
        assert trimmed == "First sentence here."

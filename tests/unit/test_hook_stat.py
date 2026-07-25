"""Tests for hook_stat.py — hero-number extraction for the hook slide."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.hook_stat import HookStat, extract_hook_stat_node


def _mock_llm_instance(stat: HookStat) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.return_value = stat
    mock_llm.invoke.return_value = stat
    return mock_llm


class TestExtractHookStatNode:
    def test_skips_when_no_content(self):
        result = extract_hook_stat_node({"deep_analysis": {}})
        assert result["hook_stat"] == {}
        assert result["current_step"] == "hook_stat_skipped"

    def test_extracts_value_and_label(self):
        stat = HookStat(value="4", label="denoising steps vs 50 in prior work")
        with patch("app.agents.nodes.hook_stat.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(stat)
            MockLLM.return_value = mock_instance

            result = extract_hook_stat_node({
                "deep_analysis": {"breakthroughs": "Only 4 denoising steps needed.", "quantitative_results": []},
            })

        assert result["hook_stat"]["value"] == "4"
        assert result["current_step"] == "hook_stat_extracted"

    def test_llm_failure_returns_empty_gracefully(self):
        with patch("app.agents.nodes.hook_stat.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = extract_hook_stat_node({
                "deep_analysis": {"breakthroughs": "text", "quantitative_results": []},
            })

        assert result["hook_stat"] == {}
        assert result["current_step"] == "hook_stat_skipped"

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        from app.agents.nodes import hook_stat as hook_stat_module

        monkeypatch.setattr(hook_stat_module.settings, "llm_schema_max_retries", 2)
        stat = HookStat(value="4", label="denoising steps vs 50 in prior work")
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), stat]
        with patch("app.agents.nodes.hook_stat.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = extract_hook_stat_node({
                "deep_analysis": {"breakthroughs": "Only 4 denoising steps needed.", "quantitative_results": []},
            })

        assert result["current_step"] == "hook_stat_extracted"
        assert result["hook_stat"]["value"] == "4"
        assert mock_llm.call_count == 2

    def test_label_is_latex_normalized(self):
        stat = HookStat(value="10x", label="faster via $O(n \\log n)$ scaling")
        with patch("app.agents.nodes.hook_stat.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(stat)
            MockLLM.return_value = mock_instance

            result = extract_hook_stat_node({
                "deep_analysis": {"breakthroughs": "text", "quantitative_results": []},
            })

        assert "$" not in result["hook_stat"]["label"]

"""Tests for ablation_spec.py — component dependency chips."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.ablation_spec import (
    AblationComponent,
    AblationSpecExtraction,
    extract_ablation_spec_node,
)


def _mock_llm_instance(extraction: AblationSpecExtraction) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.return_value = extraction
    mock_llm.invoke.return_value = extraction
    return mock_llm


class TestExtractAblationSpecNode:
    def test_skips_when_no_content(self):
        result = extract_ablation_spec_node({"deep_analysis": {}})
        assert result["ablation_components"] == []
        assert result["ablation_chips_html"] == ""
        assert result["current_step"] == "ablation_spec_skipped"

    def test_extracts_validated_and_unvalidated_components(self):
        extraction = AblationSpecExtraction(components=[
            AblationComponent(name="Sparse Routing", validated=True),
            AblationComponent(name="Positional Encoding", validated=False),
        ])
        with patch("app.agents.nodes.ablation_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_ablation_spec_node({
                "deep_analysis": {
                    "methodology": "Uses sparse routing and positional encoding.",
                    "ablation_highlights": "Removing sparse routing drops accuracy by 4%.",
                },
            })

        assert len(result["ablation_components"]) == 2
        assert result["current_step"] == "ablation_spec_extracted"

    def test_chips_html_marks_unvalidated_components_greyed_out(self):
        extraction = AblationSpecExtraction(components=[
            AblationComponent(name="Validated Part", validated=True),
            AblationComponent(name="Untested Part", validated=False),
        ])
        with patch("app.agents.nodes.ablation_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_ablation_spec_node({
                "deep_analysis": {"methodology": "text", "ablation_highlights": "text"},
            })

        html = result["ablation_chips_html"]
        assert "Validated Part" in html
        assert "Untested Part" in html
        assert "Not ablated" in html
        assert "✓ Validated" in html

    def test_caps_at_six_components(self):
        extraction = AblationSpecExtraction(
            components=[AblationComponent(name=f"Part {i}", validated=True) for i in range(10)]
        )
        with patch("app.agents.nodes.ablation_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_ablation_spec_node({
                "deep_analysis": {"methodology": "text", "ablation_highlights": "text"},
            })

        assert len(result["ablation_components"]) == 6

    def test_llm_failure_returns_empty_gracefully(self):
        with patch("app.agents.nodes.ablation_spec.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = extract_ablation_spec_node({
                "deep_analysis": {"methodology": "text", "ablation_highlights": "text"},
            })

        assert result["ablation_components"] == []
        assert result["current_step"] == "ablation_spec_skipped"

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        from app.agents.nodes import ablation_spec as ablation_spec_module

        monkeypatch.setattr(ablation_spec_module.settings, "llm_schema_max_retries", 2)
        extraction = AblationSpecExtraction(components=[
            AblationComponent(name="Recovered Part", validated=True),
        ])
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), extraction]
        with patch("app.agents.nodes.ablation_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = extract_ablation_spec_node({
                "deep_analysis": {"methodology": "text", "ablation_highlights": "text"},
            })

        assert result["current_step"] == "ablation_spec_extracted"
        assert len(result["ablation_components"]) == 1
        assert mock_llm.call_count == 2

    def test_empty_components_list_treated_as_skipped(self):
        extraction = AblationSpecExtraction(components=[])
        with patch("app.agents.nodes.ablation_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_ablation_spec_node({
                "deep_analysis": {"methodology": "text", "ablation_highlights": "text"},
            })

        assert result["current_step"] == "ablation_spec_skipped"

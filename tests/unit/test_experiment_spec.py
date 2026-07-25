"""Tests for experiment_spec.py — structured spec-card extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.experiment_spec import NOT_REPORTED, ExperimentSpec, extract_experiment_spec_node


def _mock_llm_instance(spec: ExperimentSpec) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.return_value = spec
    mock_llm.invoke.return_value = spec
    return mock_llm


class TestExtractExperimentSpecNode:
    def test_skips_when_no_experiment_setup(self):
        result = extract_experiment_spec_node({"deep_analysis": {}})
        assert result["experiment_spec"] == {}
        assert result["current_step"] == "experiment_spec_skipped"

    def test_extracts_all_four_fields(self):
        spec = ExperimentSpec(dataset="ImageNet", baselines="ResNet-50", metrics="Top-1 Acc", compute="8xA100")
        with patch("app.agents.nodes.experiment_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(spec)
            MockLLM.return_value = mock_instance

            result = extract_experiment_spec_node({
                "deep_analysis": {"experiment_setup": "Trained on ImageNet using 8xA100 GPUs..."},
            })

        assert result["experiment_spec"]["dataset"] == "ImageNet"
        assert result["experiment_spec"]["compute"] == "8xA100"
        assert result["current_step"] == "experiment_spec_extracted"

    def test_not_reported_fields_preserved_verbatim(self):
        spec = ExperimentSpec(dataset="ImageNet", baselines=NOT_REPORTED, metrics="Top-1", compute=NOT_REPORTED)
        with patch("app.agents.nodes.experiment_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(spec)
            MockLLM.return_value = mock_instance

            result = extract_experiment_spec_node({
                "deep_analysis": {"experiment_setup": "Trained on ImageNet."},
            })

        assert result["experiment_spec"]["baselines"] == NOT_REPORTED
        assert result["experiment_spec"]["compute"] == NOT_REPORTED

    def test_llm_failure_returns_empty_gracefully(self):
        with patch("app.agents.nodes.experiment_spec.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = extract_experiment_spec_node({
                "deep_analysis": {"experiment_setup": "Some setup text."},
            })

        assert result["experiment_spec"] == {}
        assert result["current_step"] == "experiment_spec_skipped"

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        from app.agents.nodes import experiment_spec as experiment_spec_module

        monkeypatch.setattr(experiment_spec_module.settings, "llm_schema_max_retries", 2)
        spec = ExperimentSpec(dataset="ImageNet", baselines="ResNet-50", metrics="Top-1 Acc", compute="8xA100")
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), spec]
        with patch("app.agents.nodes.experiment_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = extract_experiment_spec_node({
                "deep_analysis": {"experiment_setup": "Trained on ImageNet using 8xA100 GPUs..."},
            })

        assert result["current_step"] == "experiment_spec_extracted"
        assert result["experiment_spec"]["dataset"] == "ImageNet"
        assert mock_llm.call_count == 2

    def test_latex_normalized_in_extracted_fields(self):
        spec = ExperimentSpec(dataset="$O(n^2)$ synthetic set", baselines=NOT_REPORTED, metrics=NOT_REPORTED, compute=NOT_REPORTED)
        with patch("app.agents.nodes.experiment_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(spec)
            MockLLM.return_value = mock_instance

            result = extract_experiment_spec_node({
                "deep_analysis": {"experiment_setup": "Some setup text."},
            })

        assert "$" not in result["experiment_spec"]["dataset"]

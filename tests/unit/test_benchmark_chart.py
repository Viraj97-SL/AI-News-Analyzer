"""
Tests for benchmark_chart_node — metric extraction (mocked) + chart rendering.

Covers:
  - n == 1, no prior-SOTA value  -> stat card path
  - n == 1, with prior-SOTA value -> bar chart path (not degenerate to a sliver)
  - n == 5                        -> original multi-bar layout
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.benchmark_chart import (
    BenchmarkExtraction,
    BenchmarkMetric,
    benchmark_chart_node,
)

MIN_PNG_SIZE_BYTES = 3_000


def _mock_llm(extraction: BenchmarkExtraction) -> MagicMock:
    mock_chain = MagicMock()
    mock_chain.return_value = extraction          # callable path
    mock_chain.invoke.return_value = extraction   # .invoke() path
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_chain
    return mock_llm_instance


def _run_node(extraction: BenchmarkExtraction, tmp_path, run_id: str = "testrun") -> dict:
    with patch("app.agents.nodes.benchmark_chart.ChatGoogleGenerativeAI") as mock_llm_cls, \
         patch("app.agents.nodes.benchmark_chart.OUTPUT_DIR", tmp_path):
        mock_llm_cls.return_value = _mock_llm(extraction)
        result = benchmark_chart_node({
            "deep_analysis": {"breakthroughs": "Achieves 89.2% on MMLU."},
            "run_id": run_id,
        })
    return result


class TestBenchmarkChartNodeSingleMetricNoSota:
    def test_generates_chart_path(self, tmp_path):
        extraction = BenchmarkExtraction(
            metrics=[BenchmarkMetric(metric_name="MMLU", new_paper_value=89.2, unit="%")]
        )
        result = _run_node(extraction, tmp_path)
        assert result["current_step"] == "benchmark_chart_generated"
        assert result["benchmark_chart_path"]

    def test_png_written_and_non_trivial(self, tmp_path):
        extraction = BenchmarkExtraction(
            metrics=[BenchmarkMetric(metric_name="MMLU", new_paper_value=89.2, unit="%")]
        )
        result = _run_node(extraction, tmp_path)
        chart_path = result["benchmark_chart_path"]
        assert chart_path
        from pathlib import Path
        assert Path(chart_path).stat().st_size > MIN_PNG_SIZE_BYTES

    def test_metrics_round_trip(self, tmp_path):
        extraction = BenchmarkExtraction(
            metrics=[BenchmarkMetric(metric_name="MMLU", new_paper_value=89.2, unit="%")]
        )
        result = _run_node(extraction, tmp_path)
        assert result["benchmark_metrics"] == [
            {"metric_name": "MMLU", "new_paper_value": 89.2, "prior_sota_value": None, "unit": "%"}
        ]


class TestBenchmarkChartNodeSingleMetricWithSota:
    def test_generates_chart_path(self, tmp_path):
        extraction = BenchmarkExtraction(
            metrics=[
                BenchmarkMetric(
                    metric_name="MMLU", new_paper_value=89.2, prior_sota_value=86.1, unit="%"
                )
            ]
        )
        result = _run_node(extraction, tmp_path)
        assert result["current_step"] == "benchmark_chart_generated"

        from pathlib import Path
        assert Path(result["benchmark_chart_path"]).stat().st_size > MIN_PNG_SIZE_BYTES

    def test_metrics_round_trip_with_sota(self, tmp_path):
        extraction = BenchmarkExtraction(
            metrics=[
                BenchmarkMetric(
                    metric_name="MMLU", new_paper_value=89.2, prior_sota_value=86.1, unit="%"
                )
            ]
        )
        result = _run_node(extraction, tmp_path)
        assert result["benchmark_metrics"][0]["prior_sota_value"] == 86.1


class TestBenchmarkChartNodeFiveMetrics:
    def _five_metrics(self) -> BenchmarkExtraction:
        return BenchmarkExtraction(
            metrics=[
                BenchmarkMetric(
                    metric_name="MMLU", new_paper_value=89.2, prior_sota_value=86.1, unit="%"
                ),
                BenchmarkMetric(
                    metric_name="HellaSwag", new_paper_value=95.1, prior_sota_value=93.3, unit="%"
                ),
                BenchmarkMetric(
                    metric_name="ARC", new_paper_value=92.4, prior_sota_value=89.7, unit="%"
                ),
                BenchmarkMetric(metric_name="GSM8K", new_paper_value=88.0, unit="%"),
                BenchmarkMetric(metric_name="HumanEval", new_paper_value=71.5, unit="%"),
            ]
        )

    def test_generates_chart_path(self, tmp_path):
        result = _run_node(self._five_metrics(), tmp_path)
        assert result["current_step"] == "benchmark_chart_generated"

    def test_png_written_and_non_trivial(self, tmp_path):
        result = _run_node(self._five_metrics(), tmp_path)
        from pathlib import Path
        assert Path(result["benchmark_chart_path"]).stat().st_size > MIN_PNG_SIZE_BYTES

    def test_all_five_metrics_round_trip(self, tmp_path):
        result = _run_node(self._five_metrics(), tmp_path)
        assert len(result["benchmark_metrics"]) == 5
        assert result["benchmark_metrics"][3]["metric_name"] == "GSM8K"


class TestBenchmarkChartNodeEdgeCases:
    def test_skips_when_no_breakthroughs(self):
        result = benchmark_chart_node({"deep_analysis": {}, "run_id": "x"})
        assert result["current_step"] == "benchmark_skipped"
        assert result["benchmark_chart_path"] == ""

    def test_skips_when_extraction_returns_empty(self, tmp_path):
        extraction = BenchmarkExtraction(metrics=[])
        result = _run_node(extraction, tmp_path)
        assert result["current_step"] == "benchmark_skipped"
        assert result["benchmark_metrics"] == []

    def test_skips_gracefully_on_llm_failure(self, tmp_path):
        with patch("app.agents.nodes.benchmark_chart.ChatGoogleGenerativeAI") as mock_llm_cls, \
             patch("app.agents.nodes.benchmark_chart.OUTPUT_DIR", tmp_path):
            mock_chain = MagicMock()
            mock_chain.side_effect = RuntimeError("API down")
            mock_chain.invoke.side_effect = RuntimeError("API down")
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_chain
            mock_llm_cls.return_value = mock_instance

            result = benchmark_chart_node({
                "deep_analysis": {"breakthroughs": "Some results."},
                "run_id": "x",
            })

        assert result["current_step"] == "benchmark_skipped"
        assert result["benchmark_chart_path"] == ""

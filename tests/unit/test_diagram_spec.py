"""Tests for diagram_spec.py — structured architecture diagram generation + SVG rendering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.diagram_spec import (
    ArchitectureDiagramSpec,
    DiagramEdge,
    DiagramNode,
    generate_architecture_spec_node,
    render_diagram_svg,
)


def _simple_spec() -> ArchitectureDiagramSpec:
    return ArchitectureDiagramSpec(
        nodes=[
            DiagramNode(id="input", label="Input Data", kind="input"),
            DiagramNode(id="encoder", label="Encoder", kind="process"),
            DiagramNode(id="output", label="Prediction", kind="output"),
        ],
        edges=[
            DiagramEdge(source="input", target="encoder", label=""),
            DiagramEdge(source="encoder", target="output", label="logits"),
        ],
    )


class TestRenderDiagramSvg:
    def test_returns_valid_svg_wrapper(self):
        svg = render_diagram_svg(_simple_spec())
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

    def test_includes_all_node_labels(self):
        svg = render_diagram_svg(_simple_spec())
        assert "Input Data" in svg
        assert "Encoder" in svg
        assert "Prediction" in svg

    def test_includes_edge_label(self):
        svg = render_diagram_svg(_simple_spec())
        assert "logits" in svg

    def test_empty_nodes_returns_empty_string(self):
        assert render_diagram_svg(ArchitectureDiagramSpec(nodes=[], edges=[])) == ""

    def test_edge_referencing_unknown_node_is_dropped_not_crashed(self):
        spec = ArchitectureDiagramSpec(
            nodes=[DiagramNode(id="a", label="A", kind="input")],
            edges=[DiagramEdge(source="a", target="ghost", label="")],
        )
        svg = render_diagram_svg(spec)  # must not raise
        assert "<svg" in svg

    def test_wraps_to_multiple_rows_when_exceeding_max_width(self):
        many_nodes = [DiagramNode(id=f"n{i}", label=f"Node {i}", kind="process") for i in range(10)]
        spec = ArchitectureDiagramSpec(nodes=many_nodes, edges=[])
        svg = render_diagram_svg(spec, max_width=400)
        assert svg.count("<rect") == 10


class TestGenerateArchitectureSpecNode:
    def test_skipped_when_real_figure_already_present(self):
        state = {"architecture_diagram_b64": "some_base64_data", "deep_analysis": {"methodology": "x"}}
        with patch("app.agents.nodes.diagram_spec.ChatGoogleGenerativeAI") as MockLLM:
            result = generate_architecture_spec_node(state)

        assert result["architecture_spec_svg"] == ""
        assert result["current_step"] == "architecture_spec_not_needed"
        MockLLM.assert_not_called()

    def test_skipped_when_no_methodology(self):
        result = generate_architecture_spec_node({"architecture_diagram_b64": "", "deep_analysis": {}})
        assert result["architecture_spec_svg"] == ""
        assert result["current_step"] == "architecture_spec_skipped"

    def test_generates_svg_when_no_figure_and_methodology_present(self):
        spec = _simple_spec()
        mock_llm = MagicMock()
        mock_llm.return_value = spec
        mock_llm.invoke.return_value = spec

        with patch("app.agents.nodes.diagram_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = generate_architecture_spec_node({
                "architecture_diagram_b64": "",
                "deep_analysis": {"methodology": "Some methodology text.", "technical_innovation": "New idea."},
            })

        assert "<svg" in result["architecture_spec_svg"]
        assert result["current_step"] == "architecture_spec_generated"

    def test_llm_failure_returns_empty_gracefully(self):
        with patch("app.agents.nodes.diagram_spec.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = generate_architecture_spec_node({
                "architecture_diagram_b64": "",
                "deep_analysis": {"methodology": "text"},
            })

        assert result["architecture_spec_svg"] == ""
        assert result["current_step"] == "architecture_spec_skipped"

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        from app.agents.nodes import diagram_spec as diagram_spec_module

        monkeypatch.setattr(diagram_spec_module.settings, "llm_schema_max_retries", 2)
        spec = _simple_spec()
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), spec]

        with patch("app.agents.nodes.diagram_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = generate_architecture_spec_node({
                "architecture_diagram_b64": "",
                "deep_analysis": {"methodology": "Some methodology text.", "technical_innovation": "New idea."},
            })

        assert result["current_step"] == "architecture_spec_generated"
        assert "<svg" in result["architecture_spec_svg"]
        assert mock_llm.call_count == 2

    def test_whitespace_only_b64_treated_as_no_figure(self):
        spec = _simple_spec()
        mock_llm = MagicMock()
        mock_llm.return_value = spec
        mock_llm.invoke.return_value = spec

        with patch("app.agents.nodes.diagram_spec.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = generate_architecture_spec_node({
                "architecture_diagram_b64": "   ",
                "deep_analysis": {"methodology": "text"},
            })

        assert result["current_step"] == "architecture_spec_generated"

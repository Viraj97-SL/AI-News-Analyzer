"""
Tests for full_text.py — PDF download + Results/Ablation/Experiment-Setup
section extraction feeding deep_analysis_node beyond the abstract.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from app.agents.nodes.full_text import fetch_full_text_node

_SAMPLE_RAW_TEXT = (
    "Page 1 text\n"
    "1 Introduction\n"
    "Some intro text.\n"
    "4 Experimental Setup\n"
    "Dataset details here. Baselines: X, Y.\n"
    "5 Results\n"
    "We achieve 90% accuracy.\n"
    "6 Ablation Study\n"
    "Removing component A drops score by 5%.\n"
    "7 Conclusion\n"
    "We conclude."
)


def _mock_fitz_with_text(raw_text: str) -> MagicMock:
    mock_page = MagicMock()
    mock_page.get_text.return_value = raw_text
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_fitz = MagicMock()
    mock_fitz.Document.return_value = mock_doc
    return mock_fitz


class TestFetchFullTextNode:
    def test_no_paper_url_returns_empty_result(self):
        result = fetch_full_text_node({"chosen_research_paper": {}})
        assert result["full_text_available"] is False
        assert result["analysis_confidence"] == "low"
        assert result["paper_sections"] == {}

    def test_pdf_fetch_failure_falls_back_to_abstract_only(self):
        state = {"chosen_research_paper": {"url": "https://arxiv.org/abs/2607.14005"}}
        with patch("app.agents.nodes.full_text.fetch_pdf_bytes", return_value=None):
            result = fetch_full_text_node(state)

        assert result["full_text_available"] is False
        assert result["analysis_confidence"] == "low"

    def test_pdf_parse_failure_falls_back_to_abstract_only(self):
        state = {"chosen_research_paper": {"url": "https://arxiv.org/abs/2607.14005"}}
        mock_fitz = MagicMock()
        mock_fitz.Document.side_effect = RuntimeError("corrupt PDF")

        with patch("app.agents.nodes.full_text.fetch_pdf_bytes", return_value=b"%PDF-1.4..."), \
             patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = fetch_full_text_node(state)

        assert result["full_text_available"] is False
        assert result["analysis_confidence"] == "low"

    def test_no_target_sections_found_falls_back(self):
        state = {"chosen_research_paper": {"url": "https://arxiv.org/abs/2607.14005"}}
        mock_fitz = _mock_fitz_with_text("Just an abstract with no section headings at all.")

        with patch("app.agents.nodes.full_text.fetch_pdf_bytes", return_value=b"%PDF-1.4..."), \
             patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = fetch_full_text_node(state)

        assert result["full_text_available"] is False
        assert result["paper_sections"] == {}

    def test_extracts_all_three_target_sections(self):
        state = {"chosen_research_paper": {"url": "https://arxiv.org/abs/2607.14005"}}
        mock_fitz = _mock_fitz_with_text(_SAMPLE_RAW_TEXT)

        with patch("app.agents.nodes.full_text.fetch_pdf_bytes", return_value=b"%PDF-1.4..."), \
             patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = fetch_full_text_node(state)

        assert result["full_text_available"] is True
        assert result["analysis_confidence"] == "high"
        assert "Dataset details here" in result["paper_sections"]["experiment_setup"]
        assert "We achieve 90% accuracy" in result["paper_sections"]["results"]
        assert "Removing component A" in result["paper_sections"]["ablation"]

    def test_section_stops_at_next_heading_boundary(self):
        """The experiment_setup section must not bleed into the Results section that follows it."""
        state = {"chosen_research_paper": {"url": "https://arxiv.org/abs/2607.14005"}}
        mock_fitz = _mock_fitz_with_text(_SAMPLE_RAW_TEXT)

        with patch("app.agents.nodes.full_text.fetch_pdf_bytes", return_value=b"%PDF-1.4..."), \
             patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = fetch_full_text_node(state)

        assert "We achieve 90% accuracy" not in result["paper_sections"]["experiment_setup"]

    def test_section_truncated_to_configured_max_chars(self, monkeypatch):
        from app.agents.nodes import full_text as full_text_module

        monkeypatch.setattr(full_text_module.settings, "full_text_section_max_chars", 10)
        state = {"chosen_research_paper": {"url": "https://arxiv.org/abs/2607.14005"}}
        mock_fitz = _mock_fitz_with_text(_SAMPLE_RAW_TEXT)

        with patch("app.agents.nodes.full_text.fetch_pdf_bytes", return_value=b"%PDF-1.4..."), \
             patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = fetch_full_text_node(state)

        assert len(result["paper_sections"]["experiment_setup"]) <= 10

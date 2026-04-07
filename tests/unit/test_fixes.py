"""
Tests for bug fixes:
- _parse_json_tolerant: recovers summaries from truncated LLM JSON
- response.content list handling in summarizer/linkedin_gen
- PyMuPDF carousel PDF generation
"""

from __future__ import annotations

import json
import pytest

from app.agents.nodes.summarizer import _parse_json_tolerant, summarize_node, deduplicate_node


# ── _parse_json_tolerant ────────────────────────────────────────────────────

class TestParseJsonTolerant:
    def test_parses_valid_json(self):
        data = [{"headline": "A", "body": "B"}]
        assert _parse_json_tolerant(json.dumps(data)) == data

    def test_recovers_truncated_array(self):
        # Simulate LLM cutting off mid-object
        full = '[{"headline": "A", "body": "B"}, {"headline": "C", "body": "D"}, {"headline": "E"'
        result = _parse_json_tolerant(full)
        assert len(result) == 2
        assert result[0]["headline"] == "A"
        assert result[1]["headline"] == "C"

    def test_recovers_single_complete_object(self):
        truncated = '[{"headline": "Only One", "body": "complete"}, {"headline": "Cut'
        result = _parse_json_tolerant(truncated)
        assert len(result) == 1
        assert result[0]["headline"] == "Only One"

    def test_raises_on_unparseable(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_tolerant("not json at all {{{")

    def test_handles_markdown_fenced_json(self):
        import re
        raw = '```json\n[{"headline": "Test", "body": "Body"}]\n```'
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        result = _parse_json_tolerant(raw)
        assert result[0]["headline"] == "Test"


# ── response.content list handling ─────────────────────────────────────────

class TestResponseContentListHandling:
    """
    Newer langchain-google-genai returns response.content as a list of parts
    instead of a plain string. Verify the node handles both cases.
    """

    def test_summarize_node_handles_list_content(self, sample_articles):
        """summarize_node must not crash when response.content is a list."""
        from unittest.mock import MagicMock, patch

        fake_content = [{"text": '[{"headline": "H", "body": "B", "category": "LLM", "source_url": "https://x.com", "credibility_score": 0.8}]'}]
        fake_response = MagicMock()
        fake_response.content = fake_content

        scored = [{**a, "credibility_score": 0.7, "relevance_score": 0.8} for a in sample_articles]
        state = {"deduplicated_articles": scored, "feedback": ""}

        with patch("app.agents.nodes.summarizer.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = fake_response
            result = summarize_node(state)

        assert "error_log" not in result or not result.get("error_log")
        assert len(result.get("summaries", [])) == 1
        assert result["summaries"][0]["headline"] == "H"

    def test_summarize_node_handles_string_content(self, sample_articles):
        """summarize_node must still work when response.content is a plain string."""
        from unittest.mock import MagicMock, patch

        fake_response = MagicMock()
        fake_response.content = '[{"headline": "H", "body": "B", "category": "LLM", "source_url": "https://x.com", "credibility_score": 0.8}]'

        scored = [{**a, "credibility_score": 0.7, "relevance_score": 0.8} for a in sample_articles]
        state = {"deduplicated_articles": scored, "feedback": ""}

        with patch("app.agents.nodes.summarizer.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = fake_response
            result = summarize_node(state)

        assert len(result.get("summaries", [])) == 1

    def test_summarize_node_recovers_truncated_response(self, sample_articles):
        """summarize_node must recover partial summaries from truncated JSON."""
        from unittest.mock import MagicMock, patch

        truncated = '[{"headline": "H1", "body": "B1", "category": "LLM", "source_url": "https://x.com", "credibility_score": 0.8}, {"headline": "Cut'
        fake_response = MagicMock()
        fake_response.content = truncated

        scored = [{**a, "credibility_score": 0.7, "relevance_score": 0.8} for a in sample_articles]
        state = {"deduplicated_articles": scored, "feedback": ""}

        with patch("app.agents.nodes.summarizer.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = fake_response
            result = summarize_node(state)

        # Should recover the 1 complete object instead of returning 0
        assert len(result.get("summaries", [])) == 1
        assert result["summaries"][0]["headline"] == "H1"


# ── Carousel PDF via PyMuPDF ────────────────────────────────────────────────

class TestCarouselPdf:
    def test_fitz_available(self):
        """PyMuPDF must be importable (bundled codecs, no libjpeg needed)."""
        import fitz  # noqa: F401

    def test_png_to_pdf_no_codec_error(self, tmp_path):
        """Converting a PNG to PDF via fitz must not raise any codec error."""
        import fitz
        from PIL import Image

        # Create a simple 100x100 white PNG
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        png_path = tmp_path / "test.png"
        img.save(str(png_path))

        pdf_path = tmp_path / "out.pdf"
        doc = fitz.open()
        img_doc = fitz.open(str(png_path))
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        img_pdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(img_pdf)
        img_pdf.close()
        doc.save(str(pdf_path))
        doc.close()

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

"""
Tests for text_utils.py — LaTeX/Unicode normalization applied to every
LLM-produced string before it reaches a renderer.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.agents.nodes.text_utils import normalize_model_strings, normalize_text, normalize_title

# Real-world arXiv title patterns, (input, expected_output).
TITLE_CASES: list[tuple[str, str]] = [
    (
        "M$^\\text{4}$World: A Multi-view Multimodal Driving World Model",
        "M⁴World: A Multi-view Multimodal Driving World Model",
    ),
    ("Attention Is All You Need", "Attention Is All You Need"),
    ("GPT-4 Technical Report", "GPT-4 Technical Report"),
    (
        "$O(n^2)$ Attention is Not All You Need",
        "O(n²) Attention is Not All You Need",
    ),
    (
        "H$_2$O-GPT: Efficient Diffusion Models",
        "H₂O-GPT: Efficient Diffusion Models",
    ),
    (
        "Learning $\\mathbf{Z}$-transforms for Time Series",
        "Learning Z-transforms for Time Series",
    ),
    (
        "A Study of $\\alpha$-divergence in VAEs",
        "A Study of alpha-divergence in VAEs",
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    ),
    ("learning_rate_schedule_analysis", "learning_rate_schedule_analysis"),
    ("", ""),
]


class TestNormalizeText:
    @pytest.mark.parametrize("raw, expected", TITLE_CASES)
    def test_real_arxiv_title_patterns(self, raw, expected):
        assert normalize_text(raw) == expected

    def test_normalize_title_is_alias(self):
        raw = "M$^\\text{4}$World"
        assert normalize_title(raw) == normalize_text(raw)

    def test_no_dollar_signs_survive(self):
        assert "$" not in normalize_text("This costs $5 in $\\text{compute}$")

    def test_no_text_wrapper_survives(self):
        assert "\\text" not in normalize_text("M$^\\text{4}$World")

    def test_braced_superscript_without_unicode_mapping_drops_caret_and_braces(self):
        result = normalize_text("Learning $x^{k}$ Representations")
        assert "^" not in result
        assert "{" not in result and "}" not in result

    def test_idempotent(self):
        raw = "M$^\\text{4}$World"
        once = normalize_text(raw)
        twice = normalize_text(once)
        assert once == twice

    def test_none_like_empty_string_passthrough(self):
        assert normalize_text("") == ""

    def test_collapses_whitespace_left_by_removed_markup(self):
        result = normalize_text("A  $  $  B")
        assert "  " not in result


class _DummyModel(BaseModel):
    title: str
    tags: list[str]
    score: int


class TestNormalizeModelStrings:
    def test_normalizes_str_field(self):
        model = _DummyModel(title="M$^\\text{4}$World", tags=[], score=1)
        result = normalize_model_strings(model)
        assert result.title == "M⁴World"

    def test_normalizes_list_of_str_field(self):
        model = _DummyModel(title="x", tags=["$O(n^2)$ scaling", "plain tag"], score=1)
        result = normalize_model_strings(model)
        assert result.tags == ["O(n²) scaling", "plain tag"]

    def test_leaves_non_string_fields_untouched(self):
        model = _DummyModel(title="x", tags=[], score=42)
        result = normalize_model_strings(model)
        assert result.score == 42

    def test_returns_same_model_class(self):
        model = _DummyModel(title="x", tags=[], score=1)
        result = normalize_model_strings(model)
        assert isinstance(result, _DummyModel)

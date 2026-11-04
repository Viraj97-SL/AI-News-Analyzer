"""Tests for quota-aware research model selection (D22)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core import models

SETTINGS = SimpleNamespace(
    research_deep_models=["gemini-4-argon", "gemini-3.1-pro-preview", "gemini-3.5-flash"],
    research_fast_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
)


@pytest.fixture(autouse=True)
def _isolate():
    models.reset_resolved_models()
    with patch.object(models, "get_settings", return_value=SETTINGS):
        yield
    models.reset_resolved_models()


def _client(available: set[str]):
    client = MagicMock()

    def generate(model, contents):
        if model not in available:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return SimpleNamespace(text="OK")

    client.models.generate_content.side_effect = generate
    return client


def test_unresolved_returns_first_preference():
    assert models.research_model("deep") == "gemini-4-argon"
    assert models.research_model("fast") == "gemini-3.5-flash-lite"


def test_resolve_falls_back_past_quota_errors():
    out = models.resolve_research_models(_client({"gemini-3.5-flash", "gemini-3.5-flash-lite"}))
    assert out == {"deep": "gemini-3.5-flash", "fast": "gemini-3.5-flash-lite"}
    assert models.research_model("deep") == "gemini-3.5-flash"


def test_resolve_prefers_top_model_when_available():
    out = models.resolve_research_models(_client({"gemini-4-argon", "gemini-3.5-flash-lite"}))
    assert out["deep"] == "gemini-4-argon"


def test_resolve_keeps_first_preference_when_nothing_answers():
    out = models.resolve_research_models(_client(set()))
    assert out == {"deep": "gemini-4-argon", "fast": "gemini-3.5-flash-lite"}


def test_empty_preferences_raise():
    empty = SimpleNamespace(research_deep_models=[], research_fast_models=[])
    with patch.object(models, "get_settings", return_value=empty), pytest.raises(ValueError):
        models.research_model("deep")


def test_research_nodes_have_no_hardcoded_gemini_names():
    """Guard: research nodes must go through research_model(), not literal model names."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "app" / "agents"
    news_only = {"summarizer.py", "linkedin_gen.py", "credibility.py", "image_gen.py", "graph.py"}
    offenders = [
        str(p)
        for p in root.rglob("*.py")
        if p.name not in news_only and re.search(r'model="gemini-', p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_news_model_defaults_are_not_retired_gemini_2_5():
    """Gemini 2.5 returns 404 for new keys — defaults must point at a live generation."""
    from app.core.config import Settings

    for field in ("model_classifier", "model_summarizer", "model_analyzer"):
        assert not Settings.model_fields[field].default.startswith("gemini-2")

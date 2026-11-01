"""Tests for the optional AI cover illustration (decisions D8/D9)."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.agents.nodes import cover_image as ci

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"


def _fake_client(image_bytes=PNG_BYTES):
    part_text = SimpleNamespace(inline_data=None, text="here you go")
    part_img = SimpleNamespace(
        inline_data=SimpleNamespace(data=image_bytes, mime_type="image/png"), text=None
    )
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part_text, part_img]))]
    )
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


STATE = {
    "run_id": "r1",
    "chosen_research_paper": {
        "title": "Fresh Context: Verifier-Gated Retrieval",
        "trend_tags": ["rag"],
    },
    "deep_analysis": {"technical_innovation": "A learned freshness score for cached tool output."},
}


def test_build_cover_prompt_forbids_text_and_uses_concept():
    prompt = ci.build_cover_prompt(STATE["chosen_research_paper"], STATE["deep_analysis"])
    assert "Fresh Context" in prompt
    assert "freshness score" in prompt
    assert "no text" in prompt.lower()
    assert "no logos" in prompt.lower()


def test_extract_image_bytes_picks_inline_image_part():
    assert ci.extract_image_bytes(_fake_client().models.generate_content.return_value) == PNG_BYTES


def test_extract_image_bytes_none_when_no_image():
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
    assert ci.extract_image_bytes(response) is None
    assert ci.extract_image_bytes(SimpleNamespace(candidates=[])) is None


def test_node_disabled_by_default_makes_no_call():
    client = _fake_client()
    with patch.object(
        ci, "_settings", return_value=SimpleNamespace(research_cover_image_enabled=False)
    ):
        assert ci.cover_image_node(STATE, client=client) == {}
    client.models.generate_content.assert_not_called()


def test_node_enabled_returns_b64_and_writes_file(tmp_path):
    settings = SimpleNamespace(research_cover_image_enabled=True, research_cover_image_model="m")
    with (
        patch.object(ci, "_settings", return_value=settings),
        patch.object(ci, "OUTPUT_DIR", tmp_path),
    ):
        out = ci.cover_image_node(STATE, client=_fake_client())
    assert base64.b64decode(out["cover_image_b64"]) == PNG_BYTES
    assert (tmp_path / "cover_art_r1.png").read_bytes() == PNG_BYTES


def test_node_failure_is_non_fatal(tmp_path):
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("quota exceeded")
    settings = SimpleNamespace(research_cover_image_enabled=True, research_cover_image_model="m")
    with (
        patch.object(ci, "_settings", return_value=settings),
        patch.object(ci, "OUTPUT_DIR", tmp_path),
    ):
        out = ci.cover_image_node(STATE, client=client)
    assert "cover_image_b64" not in out
    assert "quota" in out["error_log"][0]


def test_node_skips_without_paper():
    settings = SimpleNamespace(research_cover_image_enabled=True, research_cover_image_model="m")
    with patch.object(ci, "_settings", return_value=settings):
        assert ci.cover_image_node({"run_id": "r"}, client=_fake_client()) == {}


def test_cover_image_enabled_by_default_in_settings():
    """D21: user opted in to cover generation."""
    from app.core.config import Settings

    assert Settings.model_fields["research_cover_image_enabled"].default is True


def test_node_failure_message_is_truncated(tmp_path):
    client = MagicMock()
    long_error = "429 RESOURCE_EXHAUSTED " + "x" * 5000
    client.models.generate_content.side_effect = RuntimeError(long_error)
    settings = SimpleNamespace(research_cover_image_enabled=True, research_cover_image_model="m")
    with (
        patch.object(ci, "_settings", return_value=settings),
        patch.object(ci, "OUTPUT_DIR", tmp_path),
    ):
        out = ci.cover_image_node(STATE, client=client)
    assert len(out["error_log"][0]) < 400
    assert "429" in out["error_log"][0]

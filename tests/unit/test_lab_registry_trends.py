"""Tests for superlab detection and AI-trend tagging (research pipeline v2, phases 2-3)."""

from __future__ import annotations

import pytest

from app.agents.nodes.lab_registry import LAB_TIER_BONUS, detect_lab, lab_bonus
from app.agents.nodes.trends import (
    TREND_TAXONOMY,
    tag_trends,
    trend_bonus,
    trend_hashtags,
    trend_labels,
)

# ── lab registry ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Google DeepMind", "Google DeepMind"),
        ("google", "Google DeepMind"),
        ("Google Research, Mountain View", "Google DeepMind"),
        ("OpenAI", "OpenAI"),
        ("Anthropic", "Anthropic"),
        ("Meta AI (FAIR)", "Meta"),
        ("facebook", "Meta"),
        ("Microsoft Research Asia", "Microsoft"),
        ("NVIDIA", "NVIDIA"),
        ("Qwen Team, Alibaba Group", "Alibaba Qwen"),
        ("deepseek-ai", "DeepSeek"),
        ("Stanford University", "Stanford"),
        ("Allen Institute for AI", "AI2"),
    ],
)
def test_detect_lab_known_names(text, expected):
    lab = detect_lab(text)
    assert lab is not None
    assert lab.name == expected


def test_detect_lab_superlab_tier_is_one():
    assert detect_lab("OpenAI").tier == 1
    assert detect_lab("Stanford University").tier == 2


@pytest.mark.parametrize(
    "text", ["metadata-driven learning", "meta-learning for few-shot", "Georgia Tech", "", None]
)
def test_detect_lab_no_false_positives(text):
    assert detect_lab(text) is None


def test_detect_lab_prefers_superlab_when_multiple_present():
    lab = detect_lab("Stanford University", "Work done while at Google DeepMind")
    assert lab.name == "Google DeepMind"


def test_lab_bonus_values():
    assert lab_bonus(detect_lab("Anthropic")) == LAB_TIER_BONUS[1]
    assert lab_bonus(detect_lab("MIT CSAIL")) == LAB_TIER_BONUS[2]
    assert lab_bonus(None) == 0.0
    # Decision D4: a superlab boosts but cannot single-handedly beat a +3.0 manual pick
    assert LAB_TIER_BONUS[1] < 3.0


# ── trends ───────────────────────────────────────────────────────────────────


def test_tag_trends_detects_core_trends():
    text = (
        "We present an MCP server benchmark for tool calling, combined with "
        "retrieval-augmented generation and a multi-agent orchestration layer."
    )
    tags = tag_trends(text)
    assert {"mcp_tool_use", "rag", "multi_agent"} <= set(tags)


def test_tag_trends_case_sensitive_acronyms():
    assert "rag" not in tag_trends("the drag coefficient of a rag doll")
    assert "rag" in tag_trends("A new RAG pipeline")
    assert "mcp_tool_use" not in tag_trends("mcp joint of the hand")


def test_tag_trends_reasoning_and_efficiency():
    tags = tag_trends(
        "Test-time compute scaling with GRPO; we also apply post-training quantization."
    )
    assert "reasoning" in tags
    assert "efficient_models" in tags


def test_tag_trends_empty_text():
    assert tag_trends("") == []
    assert tag_trends(None) == []


def test_tag_trends_order_follows_taxonomy_and_unique():
    tags = tag_trends("RAG RAG multi-agent RAG")
    keys = list(TREND_TAXONOMY)
    assert tags == sorted(set(tags), key=keys.index)


def test_trend_bonus_capped():
    assert trend_bonus([]) == 0.0
    assert trend_bonus(["rag"]) == 0.5
    assert trend_bonus(["rag", "multi_agent", "reasoning", "mcp_tool_use"]) == 1.0


def test_trend_labels_and_hashtags():
    assert trend_labels(["rag"]) == ["RAG"]
    assert trend_hashtags(["rag", "multi_agent"]) == ["#RAG", "#MultiAgentSystems"]
    assert trend_labels(["unknown"]) == []


@pytest.mark.parametrize(
    "text", ["a fair comparison", "Code released under the MIT License", "apple-picking robot"]
)
def test_detect_lab_acronym_and_license_false_positives(text):
    assert detect_lab(text) is None


def test_detect_lab_acronyms_case_sensitive_match():
    assert detect_lab("Work done at FAIR").name == "Meta"
    assert detect_lab("MIT, Cambridge MA").name == "MIT"


# ── review fixes: strict free-text matching & phrase-level trends ────────────


@pytest.mark.parametrize(
    "comment",
    [
        "Code: github.com/google/jax",
        "Our agent is built on Qwen",
        "Presented at Google I/O",
        "apple",
    ],
)
def test_detect_lab_strict_ignores_ambiguous_free_text(comment):
    assert detect_lab(comment, strict=True) is None


def test_detect_lab_strict_accepts_explicit_affiliations():
    assert detect_lab("Work done at Google DeepMind", strict=True).name == "Google DeepMind"
    assert detect_lab("Equal contribution; OpenAI", strict=True).name == "OpenAI"
    assert detect_lab("Microsoft Research Asia", strict=True).name == "Microsoft"


@pytest.mark.parametrize(
    ("text", "unwanted"),
    [
        ("Accurate alignment of satellite images", "alignment_safety"),
        ("A mechanistic model of fluid physics", "alignment_safety"),
        ("Tree pruning heuristics for chess search", "efficient_models"),
        ("Multimodal transport logistics planning", "multimodal"),
    ],
)
def test_tag_trends_avoids_generic_word_false_positives(text, unwanted):
    assert unwanted not in tag_trends(text)


def test_tag_trends_phrase_level_positives():
    assert "alignment_safety" in tag_trends(
        "We study LLM alignment via mechanistic interpretability"
    )
    assert "multimodal" in tag_trends("A multimodal large language model for charts")

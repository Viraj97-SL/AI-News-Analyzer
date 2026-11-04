"""
AI trend taxonomy and deterministic trend tagging (decision D6: no extra LLM call).

Tags feed three places: the ranker's trend bonus, the "trend" chip on the cover
slide, and topic hashtags for the LinkedIn post. Edit `TREND_TAXONOMY` to track
new trends; dict order is the display/priority order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Trend:
    label: str
    hashtag: str
    patterns: tuple[re.Pattern[str], ...]


def _ci(*phrases: str) -> tuple[re.Pattern[str], ...]:
    """Case-insensitive whole-phrase patterns."""
    return tuple(re.compile(rf"(?<![\w-]){re.escape(p)}(?![\w])", re.IGNORECASE) for p in phrases)


def _cs(*acronyms: str) -> tuple[re.Pattern[str], ...]:
    """Case-sensitive acronyms (so 'rag doll' never matches RAG)."""
    return tuple(re.compile(rf"\b{re.escape(a)}\b") for a in acronyms)


TREND_TAXONOMY: dict[str, Trend] = {
    "mcp_tool_use": Trend(
        "MCP & Tool Use",
        "#MCP",
        _cs("MCP")
        + _ci(
            "model context protocol",
            "tool calling",
            "tool use",
            "function calling",
            "tool-augmented",
        ),
    ),
    "rag": Trend(
        "RAG",
        "#RAG",
        _cs("RAG", "GraphRAG")
        + _ci("retrieval-augmented", "retrieval augmented", "graph rag", "agentic retrieval"),
    ),
    "multi_agent": Trend(
        "Multi-Agent Systems",
        "#MultiAgentSystems",
        _ci(
            "multi-agent",
            "multi agent",
            "multiagent",
            "agent collaboration",
            "agent orchestration",
            "agent swarm",
        ),
    ),
    "ai_agents": Trend(
        "AI Agents",
        "#AIAgents",
        _ci(
            "llm agent",
            "llm agents",
            "autonomous agent",
            "autonomous agents",
            "agentic",
            "computer use",
            "computer-use",
            "web agent",
            "web agents",
            "coding agent",
            "coding agents",
            "software engineering agent",
        ),
    ),
    "agent_evals": Trend(
        "Evals & Benchmarks",
        "#LLMEvals",
        _ci(
            "llm-as-a-judge",
            "llm as a judge",
            "agent evaluation",
            "agent benchmark",
            "evaluation harness",
            "benchmark contamination",
        ),
    ),
    "reasoning": Trend(
        "Reasoning & Test-Time Compute",
        "#Reasoning",
        _cs("GRPO", "RLVR")
        + _ci(
            "chain-of-thought",
            "chain of thought",
            "test-time compute",
            "test-time scaling",
            "inference-time scaling",
            "reasoning model",
            "reasoning models",
            "verifiable rewards",
            "long reasoning",
        ),
    ),
    "efficient_models": Trend(
        "Efficient & Small Models",
        "#EfficientAI",
        _cs("MoE")
        + _ci(
            "post-training quantization",
            "weight quantization",
            "quantized llm",
            "quantized model",
            "knowledge distillation",
            "model distillation",
            "small language model",
            "mixture-of-experts",
            "mixture of experts",
            "speculative decoding",
            "kv cache",
            "kv-cache",
            "model pruning",
            "structured pruning",
        ),
    ),
    "multimodal": Trend(
        "Multimodal & World Models",
        "#Multimodal",
        _cs("VLM", "VLA")
        + _ci(
            "vision-language",
            "multimodal llm",
            "multimodal large language model",
            "multimodal model",
            "multimodal models",
            "world model",
            "world models",
            "video generation",
            "vision-language-action",
        ),
    ),
    "alignment_safety": Trend(
        "Alignment & Interpretability",
        "#AISafety",
        _ci(
            "ai alignment",
            "llm alignment",
            "model alignment",
            "value alignment",
            "preference alignment",
            "interpretability",
            "mechanistic interpretability",
            "sparse autoencoder",
            "jailbreak",
            "reward hacking",
            "red teaming",
            "red-teaming",
            "ai safety",
        ),
    ),
    "long_context_memory": Trend(
        "Long Context & Memory",
        "#LongContext",
        _ci(
            "long-context",
            "long context",
            "context window",
            "agent memory",
            "memory-augmented",
            "long-term memory",
        ),
    ),
}

_BONUS_PER_TREND = 0.5
_MAX_TREND_BONUS = 1.0


def tag_trends(text: str | None) -> list[str]:
    """Return matching trend keys in taxonomy order (unique)."""
    if not text:
        return []
    return [
        key for key, trend in TREND_TAXONOMY.items() if any(p.search(text) for p in trend.patterns)
    ]


def trend_bonus(tags: list[str]) -> float:
    return min(_MAX_TREND_BONUS, _BONUS_PER_TREND * len(tags))


def trend_labels(tags: list[str]) -> list[str]:
    return [TREND_TAXONOMY[t].label for t in tags if t in TREND_TAXONOMY]


def trend_hashtags(tags: list[str]) -> list[str]:
    return [TREND_TAXONOMY[t].hashtag for t in tags if t in TREND_TAXONOMY]

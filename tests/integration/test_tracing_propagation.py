"""Integration: run config callbacks reach LLM calls made inside sync graph nodes.

The research nodes call `(prompt | llm).invoke(...)` without passing config.
UsageTracker only works if LangGraph propagates the invoke-time config to
those nested calls — this test pins that behaviour.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.core.tracing import UsageTracker, build_research_run_config


class _S(TypedDict):
    out: str


def _make_graph():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "gemini-2.5-flash"},
    )
    llm = GenericFakeChatModel(messages=iter([msg]))

    def node(state: _S) -> dict:
        return {"out": llm.invoke("hello").content}  # no config passed, like production nodes

    g = StateGraph(_S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


def test_usage_tracker_sees_llm_calls_inside_sync_nodes():
    tracker = UsageTracker()
    config = build_research_run_config("run-x", "manual", tracker)
    result = asyncio.run(_make_graph().ainvoke({"out": ""}, config))
    assert result["out"] == "hi"
    snap = tracker.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["total_tokens"] == 15


def test_cost_limit_aborts_run_even_inside_node_fallbacks():
    """Nodes wrap LLM calls in `except Exception` — the budget stop must still abort the run."""
    import pytest

    from app.core.tracing import CostLimitExceededError

    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "gemini-2.5-flash"},
    )
    llm = GenericFakeChatModel(messages=iter([msg, msg]))
    calls = {"fallbacks": 0}

    def node(state: _S) -> dict:
        try:
            llm.invoke("first")
            llm.invoke("second")  # budget already exceeded → must not be swallowed
        except Exception:
            calls["fallbacks"] += 1
        return {"out": "degraded"}

    g = StateGraph(_S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    tracker = UsageTracker(max_cost=1000.0, max_tokens=10)
    config = build_research_run_config("run-y", "manual", tracker)

    with pytest.raises(CostLimitExceededError):
        asyncio.run(g.compile().ainvoke({"out": ""}, config))
    assert calls["fallbacks"] == 0

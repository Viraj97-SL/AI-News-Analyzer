"""Tests for app.core.tracing — LangSmith env export, run config, usage/cost tracking."""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.core.tracing import (
    CostLimitExceededError,
    UsageTracker,
    build_research_run_config,
    configure_langsmith,
    estimate_cost,
)

_LS_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
)


@pytest.fixture
def clean_env(monkeypatch):
    # configure_langsmith writes os.environ directly (monkeypatch can't see those writes),
    # so snapshot and fully restore the tracing vars — otherwise a fake key leaks into
    # later tests and LangChain starts uploading traces from the test run.
    saved = {var: os.environ.get(var) for var in _LS_VARS}
    for var in _LS_VARS:
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def _settings(tracing=True, key="lsv2_test", project="proj"):
    return SimpleNamespace(
        langsmith_tracing=tracing, langsmith_api_key=key, langsmith_project=project
    )


def _llm_result(model: str, inp: int, out: int) -> LLMResult:
    msg = AIMessage(
        content="x",
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


# ── configure_langsmith ──────────────────────────────────────────────────────


def test_configure_langsmith_exports_env_vars_when_enabled(clean_env):
    assert configure_langsmith(_settings()) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


def test_configure_langsmith_disabled_without_api_key(clean_env):
    assert configure_langsmith(_settings(key="")) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_configure_langsmith_disabled_when_flag_off(clean_env):
    assert configure_langsmith(_settings(tracing=False)) is False
    assert os.environ.get("LANGSMITH_TRACING") == "false"


def test_configure_langsmith_does_not_override_explicit_project(clean_env):
    clean_env.setenv("LANGSMITH_PROJECT", "from-env")
    configure_langsmith(_settings())
    assert os.environ["LANGSMITH_PROJECT"] == "from-env"


# ── cost estimation ──────────────────────────────────────────────────────────


def test_estimate_cost_known_model():
    # gemini-2.5-pro: $1.25 / 1M input, $10 / 1M output
    assert estimate_cost("gemini-2.5-pro", 1_000_000, 1_000_000) == pytest.approx(11.25)


def test_estimate_cost_matches_model_prefix_variants():
    assert estimate_cost("models/gemini-2.5-flash-001", 1_000_000, 0) == pytest.approx(0.30)


def test_estimate_cost_unknown_model_uses_conservative_default():
    assert estimate_cost("mystery-model", 1_000_000, 0) > 0


# ── UsageTracker ─────────────────────────────────────────────────────────────


def test_usage_tracker_accumulates_tokens_and_cost():
    tracker = UsageTracker(max_cost=100.0, max_tokens=10_000_000)
    tracker.on_llm_end(_llm_result("gemini-2.5-flash", 1000, 500), run_id=uuid.uuid4())
    tracker.on_llm_end(_llm_result("gemini-2.5-pro", 2000, 100), run_id=uuid.uuid4())

    snap = tracker.snapshot()
    assert snap["total_tokens"] == 3600
    assert snap["llm_calls"] == 2
    assert snap["by_model"]["gemini-2.5-flash"]["input_tokens"] == 1000
    assert snap["total_cost"] > 0


def test_usage_tracker_ignores_results_without_usage():
    tracker = UsageTracker()
    empty = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])
    tracker.on_llm_end(empty, run_id=uuid.uuid4())
    assert tracker.snapshot()["total_tokens"] == 0
    assert tracker.snapshot()["llm_calls"] == 1


def test_usage_tracker_blocks_new_calls_after_cost_limit():
    tracker = UsageTracker(max_cost=0.0001, max_tokens=10_000_000)
    tracker.on_llm_end(_llm_result("gemini-2.5-pro", 100_000, 100_000), run_id=uuid.uuid4())
    with pytest.raises(CostLimitExceededError):
        tracker.on_chat_model_start({}, [[]], run_id=uuid.uuid4())


def test_usage_tracker_blocks_new_calls_after_token_limit():
    tracker = UsageTracker(max_cost=1000.0, max_tokens=10)
    tracker.on_llm_end(_llm_result("gemini-2.5-flash", 50, 50), run_id=uuid.uuid4())
    with pytest.raises(CostLimitExceededError):
        tracker.on_llm_start({}, ["p"], run_id=uuid.uuid4())


def test_usage_tracker_allows_calls_under_limit():
    tracker = UsageTracker(max_cost=10.0, max_tokens=10_000)
    tracker.on_chat_model_start({}, [[]], run_id=uuid.uuid4())  # no raise


def test_usage_tracker_raises_errors_so_langchain_propagates_them():
    assert UsageTracker().raise_error is True


# ── build_research_run_config ────────────────────────────────────────────────


def test_build_research_run_config_shape():
    tracker = UsageTracker()
    cfg = build_research_run_config(
        run_id="abc-123",
        trigger_type="scheduled",
        tracker=tracker,
        extra_metadata={"manual_paper_url": "https://arxiv.org/abs/1"},
    )
    assert cfg["configurable"]["thread_id"] == "abc-123"
    assert cfg["run_name"] == "research_pipeline"
    assert {"research", "scheduled"} <= set(cfg["tags"])
    assert cfg["metadata"]["pipeline_run_id"] == "abc-123"
    assert cfg["metadata"]["manual_paper_url"] == "https://arxiv.org/abs/1"
    assert tracker in cfg["callbacks"]
    # root LangSmith run id is pinned so feedback can be attached afterwards
    assert isinstance(cfg["run_id"], uuid.UUID)


def test_build_research_run_config_without_tracker_has_no_callbacks():
    cfg = build_research_run_config(run_id="r", trigger_type="manual")
    assert cfg.get("callbacks", []) == []
    assert "manual" in cfg["tags"]


# ── send_langsmith_feedback ──────────────────────────────────────────────────


class _FakeClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def create_feedback(self, run_id, key, score):
        if self.fail:
            raise RuntimeError("network down")
        self.calls.append((run_id, key, score))


def test_send_langsmith_feedback_sends_numeric_metrics_only():
    from app.core.tracing import send_langsmith_feedback

    client = _FakeClient()
    rid = uuid.uuid4()
    sent = send_langsmith_feedback(
        rid,
        {"relevance_score": 7.5, "full_text_available": True, "verdict_rank": None, "lab": "Meta"},
        client=client,
    )
    assert sent == 2
    assert (rid, "relevance_score", 7.5) in client.calls
    assert (rid, "full_text_available", 1.0) in client.calls


def test_send_langsmith_feedback_swallows_errors():
    from app.core.tracing import send_langsmith_feedback

    assert (
        send_langsmith_feedback(uuid.uuid4(), {"relevance_score": 1}, client=_FakeClient(fail=True))
        == 0
    )

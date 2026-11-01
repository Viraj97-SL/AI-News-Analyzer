"""Tests for the research cron's observability wiring and the drift report script."""

from __future__ import annotations

import json
import uuid

from app.core.tracing import UsageTracker, build_research_run_config
from cron.research_trigger import build_initial_state, record_run_observability
from scripts.research_drift_report import main as drift_main


class _FakeClient:
    def __init__(self):
        self.keys = []

    def create_feedback(self, run_id, key, score):
        self.keys.append(key)


def test_build_initial_state_is_scheduled_and_pending():
    state = build_initial_state("r1")
    assert state["run_id"] == "r1"
    assert state["approval_status"] == "pending"
    assert state["error_log"] == []


def test_record_run_observability_writes_jsonl_and_feedback(tmp_path):
    tracker = UsageTracker()
    config = build_research_run_config("r1", "scheduled", tracker)
    client = _FakeClient()
    path = tmp_path / "runs.jsonl"

    metrics = record_run_observability(
        {"run_id": "r1", "relevance_score": 6.5},
        tracker,
        config,
        tracing_enabled=True,
        metrics_path=path,
        feedback_client=client,
    )

    saved = json.loads(path.read_text().strip())
    assert saved["run_id"] == "r1"
    assert saved["langsmith_run_id"] == str(config["run_id"])
    assert "relevance_score" in client.keys
    assert metrics["relevance_score"] == 6.5


def test_record_run_observability_skips_feedback_when_tracing_off(tmp_path):
    tracker = UsageTracker()
    config = build_research_run_config("r1", "scheduled", tracker)
    client = _FakeClient()
    record_run_observability({"run_id": "r1"}, tracker, config, False, tmp_path / "m.jsonl", client)
    assert client.keys == []


def test_drift_report_exit_codes(tmp_path, capsys):
    path = tmp_path / "runs.jsonl"
    rows = [{"run_id": str(uuid.uuid4()), "relevance_score": 5.0} for _ in range(4)]
    rows += [{"run_id": str(uuid.uuid4()), "relevance_score": 9.0} for _ in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    assert drift_main(["--path", str(path), "--baseline", "4", "--recent", "2"]) == 1
    assert "relevance_score" in capsys.readouterr().out
    assert drift_main(["--path", str(tmp_path / "missing.jsonl")]) == 0


# ── main() end-to-end with a stubbed graph ───────────────────────────────────


class _FakeGraph:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.config = result, error, None

    async def ainvoke(self, state, config):
        self.config = config
        if self.error:
            raise self.error
        return self.result


def _run_main(monkeypatch, tmp_path, graph):
    import asyncio
    from types import SimpleNamespace

    import app.agents.research_graph as rg
    import cron.research_trigger as rt

    monkeypatch.setattr(rg, "build_research_graph", lambda checkpointer=None: graph)
    monkeypatch.setattr(rt, "configure_langsmith", lambda s: False)
    monkeypatch.setattr(rt, "resolve_research_models", lambda: {"deep": "d", "fast": "f"})
    monkeypatch.setattr(rt, "DEFAULT_METRICS_PATH", tmp_path / "m.jsonl")
    monkeypatch.setattr(
        rt,
        "settings",
        SimpleNamespace(is_sqlite=True, max_cost_per_run=5.0, max_tokens_per_run=1000),
    )
    original = rt.record_run_observability
    monkeypatch.setattr(
        rt,
        "record_run_observability",
        lambda r, t, c, e: original(r, t, c, e, metrics_path=tmp_path / "m.jsonl"),
    )
    return asyncio.run(rt.main())


def test_main_success_records_metrics(monkeypatch, tmp_path):
    graph = _FakeGraph(result={"run_id": "x", "approval_status": "pending"})
    assert _run_main(monkeypatch, tmp_path, graph) == 0
    assert "research" in graph.config["tags"]
    assert (tmp_path / "m.jsonl").exists()


def test_main_failure_returns_1(monkeypatch, tmp_path):
    assert _run_main(monkeypatch, tmp_path, _FakeGraph(error=RuntimeError("boom"))) == 1


def test_main_failure_still_records_failed_metrics(monkeypatch, tmp_path):
    _run_main(monkeypatch, tmp_path, _FakeGraph(error=RuntimeError("boom")))
    record = json.loads((tmp_path / "m.jsonl").read_text().strip())
    assert record["approval_status"] == "failed"
    assert record["error_count"] == 1


def test_main_cost_limit_returns_1(monkeypatch, tmp_path):
    from app.core.tracing import CostLimitExceededError

    assert _run_main(monkeypatch, tmp_path, _FakeGraph(error=CostLimitExceededError("over"))) == 1


def test_event_loop_factory_uses_selector_on_windows(monkeypatch):
    import asyncio

    import cron.research_trigger as rt

    monkeypatch.setattr(rt.sys, "platform", "win32")
    assert rt._event_loop_factory() is asyncio.SelectorEventLoop
    monkeypatch.setattr(rt.sys, "platform", "linux")
    assert rt._event_loop_factory() is None

"""
Observability helpers for the research pipeline.

Three concerns live here:

1. **LangSmith activation** — pydantic-settings reads ``LANGSMITH_*`` from ``.env``
   into ``Settings`` but never exports them to ``os.environ``, which is the only
   place LangChain/LangGraph look. ``configure_langsmith`` bridges that gap.
2. **Run config** — ``build_research_run_config`` produces a ``RunnableConfig``
   with a pinned root run id, run name, tags and metadata. LangGraph propagates
   this config (and its callbacks) to every LLM call made inside nodes via
   context vars, so nodes need no changes to be traced and metered.
3. **Usage / cost tracking** — ``UsageTracker`` is a callback handler that sums
   tokens per model, estimates cost, and refuses to start new LLM calls once the
   configured per-run budget is exceeded (``max_cost_per_run``/``max_tokens_per_run``).
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

# USD per 1M tokens (input, output). Longest-prefix match on the model name.
# Approximate Gemini list prices (standard tier, <=200k context) — estimates for the
# cost guard and drift report, not billing. Update when Google changes pricing.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
}
# Unknown models are billed at the most expensive known rate so the cost guard
# errs on the side of stopping early rather than overspending.
_DEFAULT_PRICE: tuple[float, float] = (4.00, 18.00)
_TOKENS_PER_MTOK = 1_000_000


class CostLimitExceededError(BaseException):
    """Raised before an LLM call when the run's token or cost budget is spent.

    Deliberately a ``BaseException`` (like ``KeyboardInterrupt``): research nodes wrap
    LLM calls in ``except Exception`` and fall back to defaults, which would turn a
    hard budget stop into a degraded-but-continuing run. Callers that own the run
    (the research cron) catch it explicitly.
    """


class _LangSmithSettings(Protocol):
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str


def configure_langsmith(settings: _LangSmithSettings) -> bool:
    """Export LangSmith settings into ``os.environ``. Returns True when tracing is on.

    Explicit environment variables win over ``.env`` values (``setdefault``),
    so a deployment can override the project name without editing config.
    """
    enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key)
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"  # explicit off, even if the shell had it on
        return False
    if not enabled:
        return False  # flag on but no key: leave env untouched, tracing stays inactive

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # legacy name, still read by older libs
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    return True


def _resolve_price(model: str) -> tuple[float, float]:
    name = model.lower().removeprefix("models/")
    matches = [key for key in MODEL_PRICES_PER_MTOK if name.startswith(key)]
    if not matches:
        return _DEFAULT_PRICE
    return MODEL_PRICES_PER_MTOK[max(matches, key=len)]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost of one call."""
    in_price, out_price = _resolve_price(model)
    return (input_tokens * in_price + output_tokens * out_price) / _TOKENS_PER_MTOK


def _usage_from_result(response: LLMResult) -> list[tuple[str, int, int]]:
    """Extract (model, input_tokens, output_tokens) for each generation that reports usage."""
    usages: list[tuple[str, int, int]] = []
    for generation_list in response.generations:
        for gen in generation_list:
            message = getattr(gen, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if not usage:
                continue
            meta = getattr(message, "response_metadata", {}) or {}
            model = meta.get("model_name") or meta.get("model") or "unknown"
            usages.append(
                (model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
            )
    return usages


class UsageTracker(BaseCallbackHandler):
    """Thread-safe token/cost accumulator with a pre-call budget guard."""

    raise_error = (
        True  # let CostLimitExceededError abort the run instead of being logged and ignored
    )

    def __init__(self, max_cost: float = float("inf"), max_tokens: int = 2**62) -> None:
        super().__init__()
        self._max_cost = max_cost
        self._max_tokens = max_tokens
        self._lock = threading.Lock()
        self._by_model: dict[str, dict[str, float]] = {}
        self._llm_calls = 0

    # ── budget guard ─────────────────────────────────────────
    def _check_budget(self) -> None:
        snap = self.snapshot()
        if snap["total_cost"] > self._max_cost or snap["total_tokens"] > self._max_tokens:
            raise CostLimitExceededError(
                f"Run budget exceeded: ${snap['total_cost']:.4f} / {snap['total_tokens']} tokens "
                f"(limits ${self._max_cost} / {self._max_tokens} tokens)"
            )

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._check_budget()

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        self._check_budget()

    # ── accounting ───────────────────────────────────────────
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            usages = _usage_from_result(response)
        except Exception:  # accounting must never break the run
            usages = []
        with self._lock:
            self._llm_calls += 1
            for model, inp, out in usages:
                entry = self._by_model.setdefault(
                    model, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
                )
                entry["input_tokens"] += inp
                entry["output_tokens"] += out
                entry["cost"] += estimate_cost(model, inp, out)
                entry["calls"] += 1

    def snapshot(self) -> dict[str, Any]:
        """Immutable copy of the current totals."""
        with self._lock:
            by_model = {name: dict(vals) for name, vals in self._by_model.items()}
            calls = self._llm_calls
        total_tokens = int(sum(v["input_tokens"] + v["output_tokens"] for v in by_model.values()))
        total_cost = round(sum(v["cost"] for v in by_model.values()), 6)
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "llm_calls": calls,
            "by_model": by_model,
        }


# Metrics forwarded to LangSmith as feedback scores (numeric only; booleans → 0/1).
FEEDBACK_KEYS: tuple[str, ...] = (
    "relevance_score",
    "verdict_rank",
    "novelty",
    "hook_score_avg",
    "full_text_available",
    "figure_count",
    "slide_fill_ratio_avg",
    "error_count",
    "total_cost",
)


def send_langsmith_feedback(
    root_run_id: uuid.UUID, metrics: dict[str, Any], client: Any | None = None
) -> int:
    """Attach numeric run metrics to the LangSmith trace. Returns the number sent.

    Best-effort: observability must never fail the pipeline, so errors are swallowed
    by the caller-facing contract (logged count of 0) rather than raised.
    """
    from app.core.logging import get_logger

    log = get_logger(__name__)
    try:
        if client is None:
            from langsmith import Client

            client = Client()
        sent = 0
        for key in FEEDBACK_KEYS:
            value = metrics.get(key)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                client.create_feedback(run_id=root_run_id, key=key, score=float(value))
                sent += 1
        return sent
    except Exception as e:
        log.warning("langsmith_feedback_failed", error=str(e))
        return 0


def build_research_run_config(
    run_id: str,
    trigger_type: str,
    tracker: UsageTracker | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RunnableConfig for one research-graph invocation.

    ``run_id`` (the LangSmith root run id) is pinned to a fresh UUID so the caller
    can attach feedback scores to the trace after the graph finishes.
    """
    return {
        "configurable": {"thread_id": run_id},
        "run_id": uuid.uuid4(),
        "run_name": "research_pipeline",
        "tags": ["research", trigger_type],
        "metadata": {"pipeline": "research", "pipeline_run_id": run_id, **(extra_metadata or {})},
        "callbacks": [tracker] if tracker is not None else [],
    }

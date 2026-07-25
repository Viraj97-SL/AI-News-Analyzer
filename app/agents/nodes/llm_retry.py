"""
Shared retry wrapper for LLM calls that produce structured (Pydantic) output.

Several research-pipeline nodes call an LLM via `.with_structured_output(SomeModel)`
and previously caught the very first schema/validation failure and fell straight
through to an empty/default result — no retry, even though these failures are
often transient (a malformed field, a truncated response) rather than a
fundamental inability to answer.

This mirrors the retry-then-fallback strategy `text_budget.py` already uses for
character-budget enforcement (see `settings.carousel_text_max_retries`): retry the
same invocation up to a configurable max (`settings.llm_schema_max_retries`), with
structured logging on every failed attempt, before giving up. On final exhaustion
the last exception is re-raised so each call site's existing try/except
fallback-to-default logic fires exactly as it did before — this module only
replaces the "how many times do we try" mechanism, not the fallback contract.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import structlog

T = TypeVar("T")


def invoke_with_schema_retry(
    invoke_fn: Callable[[], T],
    max_retries: int,
    logger: structlog.stdlib.BoundLogger,
    context: str,
) -> T:
    """Call `invoke_fn` up to `max_retries` times, retrying on any exception.

    `invoke_fn` is typically `lambda: (prompt | llm).invoke({...})` for an LLM
    bound to `.with_structured_output(SomeModel)` — schema validation failures
    surface as generic exceptions from that call, same as any other LLM error.

    Logs `"llm_schema_retry"` on every failed attempt. Once `max_retries`
    attempts have all failed, re-raises the last exception so the caller's
    existing `except Exception` fallback fires unchanged.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return invoke_fn()
        except Exception as e:
            last_error = e
            logger.warning(
                "llm_schema_retry",
                context=context,
                attempt=attempt,
                max_retries=max_retries,
                error=str(e),
            )
    assert last_error is not None  # max_retries >= 1 guarantees the loop ran at least once
    raise last_error

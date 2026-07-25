"""Tests for llm_retry.py — the shared retry wrapper for `.with_structured_output()` calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agents.nodes.llm_retry import invoke_with_schema_retry


def _logger() -> MagicMock:
    return MagicMock()


class TestInvokeWithSchemaRetry:
    def test_returns_result_on_first_success_without_retrying(self):
        invoke_fn = MagicMock(return_value="ok")
        logger = _logger()

        result = invoke_with_schema_retry(invoke_fn, max_retries=2, logger=logger, context="ctx")

        assert result == "ok"
        assert invoke_fn.call_count == 1
        logger.warning.assert_not_called()

    def test_retries_after_failure_then_returns_success(self):
        invoke_fn = MagicMock(side_effect=[ValueError("schema validation failed"), "recovered"])
        logger = _logger()

        result = invoke_with_schema_retry(invoke_fn, max_retries=2, logger=logger, context="ctx")

        assert result == "recovered"
        assert invoke_fn.call_count == 2
        logger.warning.assert_called_once()
        _, kwargs = logger.warning.call_args
        assert kwargs["attempt"] == 1
        assert kwargs["max_retries"] == 2
        assert kwargs["context"] == "ctx"

    def test_falls_back_after_exhausting_max_retries(self):
        invoke_fn = MagicMock(side_effect=RuntimeError("always fails"))
        logger = _logger()

        with pytest.raises(RuntimeError, match="always fails"):
            invoke_with_schema_retry(invoke_fn, max_retries=2, logger=logger, context="ctx")

        assert invoke_fn.call_count == 2
        assert logger.warning.call_count == 2

    def test_logs_every_failed_attempt_with_increasing_attempt_number(self):
        invoke_fn = MagicMock(side_effect=RuntimeError("nope"))
        logger = _logger()

        with pytest.raises(RuntimeError):
            invoke_with_schema_retry(invoke_fn, max_retries=3, logger=logger, context="ctx")

        attempts = [call.kwargs["attempt"] for call in logger.warning.call_args_list]
        assert attempts == [1, 2, 3]

    def test_single_max_retry_means_no_retry_just_one_attempt(self):
        invoke_fn = MagicMock(side_effect=RuntimeError("boom"))
        logger = _logger()

        with pytest.raises(RuntimeError):
            invoke_with_schema_retry(invoke_fn, max_retries=1, logger=logger, context="ctx")

        assert invoke_fn.call_count == 1

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.models import resolve_research_models
from app.core.run_metrics import DEFAULT_METRICS_PATH, append_run_metrics, extract_research_metrics
from app.core.tracing import (
    CostLimitExceededError,
    UsageTracker,
    build_research_run_config,
    configure_langsmith,
    send_langsmith_feedback,
)

setup_logging()
logger = get_logger("research_cron")
settings = get_settings()


def build_initial_state(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "trigger_type": "scheduled",
        "raw_articles": [],
        "deduplicated_articles": [],
        "summaries": [],
        "newsletter_html": "",
        "linkedin_draft": "",
        "image_paths": [],
        "approval_status": "pending",
        "feedback": "",
        "error_log": [],
        "total_tokens": 0,
        "total_cost": 0.0,
        "current_step": "starting",
    }


def record_run_observability(
    result: dict[str, Any],
    tracker: UsageTracker,
    config: dict[str, Any],
    tracing_enabled: bool,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    feedback_client: Any | None = None,
) -> dict[str, Any]:
    """Persist run metrics (JSONL, drift input) and attach them to the LangSmith trace."""
    metrics = extract_research_metrics(result, tracker.snapshot())
    metrics["langsmith_run_id"] = str(config["run_id"])
    try:
        append_run_metrics(metrics_path, metrics)
    except OSError as e:
        logger.warning("run_metrics_write_failed", error=str(e))
    if tracing_enabled:
        send_langsmith_feedback(config["run_id"], metrics, client=feedback_client)
    logger.info(
        "research_run_metrics",
        run_id=metrics["run_id"],
        total_tokens=metrics["total_tokens"],
        total_cost=metrics["total_cost"],
        relevance_score=metrics["relevance_score"],
        verdict=metrics["verdict"],
    )
    return metrics


async def main() -> int:
    run_id = str(uuid.uuid4())
    logger.info("research_cron_triggered", run_id=run_id, schedule="Thursday Deep Tech")

    tracing_enabled = configure_langsmith(settings)
    tracker = UsageTracker(
        max_cost=settings.max_cost_per_run, max_tokens=settings.max_tokens_per_run
    )
    config = build_research_run_config(run_id=run_id, trigger_type="scheduled", tracker=tracker)
    logger.info("research_tracing", enabled=tracing_enabled, langsmith_run_id=str(config["run_id"]))

    try:
        from app.agents.research_graph import build_research_graph

        models = resolve_research_models()
        logger.info("research_models", **models)

        initial_state = build_initial_state(run_id)

        if settings.is_sqlite:
            from langgraph.checkpoint.memory import InMemorySaver

            graph = build_research_graph(checkpointer=InMemorySaver())
            result = await graph.ainvoke(initial_state, config)
        else:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(
                settings.langgraph_pg_uri
            ) as checkpointer:
                await checkpointer.setup()
                graph = build_research_graph(checkpointer=checkpointer)
                result = await graph.ainvoke(initial_state, config)

        record_run_observability(result, tracker, config, tracing_enabled)
        logger.info("research_cron_completed", run_id=run_id, status=result.get("approval_status"))
        return 0

    except (Exception, CostLimitExceededError) as e:
        logger.error("research_cron_failed", run_id=run_id, error=str(e), usage=tracker.snapshot())
        failed_state = {
            "run_id": run_id,
            "trigger_type": "scheduled",
            "approval_status": "failed",
            "error_log": [f"{type(e).__name__}: {e}"],
        }
        record_run_observability(failed_state, tracker, config, tracing_enabled)
        return 1


def _event_loop_factory():  # type: ignore[no-untyped-def]
    """Async psycopg (LangGraph's Postgres checkpointer) cannot run on Windows'
    default ProactorEventLoop, so use the selector loop there."""
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


if __name__ == "__main__":
    exit_code = asyncio.run(main(), loop_factory=_event_loop_factory())
    sys.exit(exit_code)

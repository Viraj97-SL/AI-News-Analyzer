"""
Railway cron job entry point.

Runs as a separate Railway service with schedule: 0 9 * * 2,4
(Tuesday & Thursday at 9 AM UTC)

IMPORTANT: This script must exit cleanly after completion.
Open DB connections will prevent Railway from marking the job as finished.
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("cron")
settings = get_settings()


async def main() -> int:
    """Trigger the pipeline and wait for completion."""
    run_id = str(uuid.uuid4())
    logger.info("cron_triggered", run_id=run_id, schedule="Tue/Thu 9AM UTC")

    try:
        if settings.is_sqlite:
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()
        else:
            from langgraph.checkpoint.postgres import AsyncPostgresSaver

            checkpointer = AsyncPostgresSaver.from_conn_string(settings.langgraph_pg_uri)
            await checkpointer.setup()

        from app.agents.graph import build_graph

        graph = build_graph(checkpointer=checkpointer)

        initial_state = {
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

        config = {"configurable": {"thread_id": run_id}}
        result = await graph.ainvoke(initial_state, config)

        logger.info(
            "cron_completed",
            run_id=run_id,
            status=result.get("approval_status"),
            step=result.get("current_step"),
            tokens=result.get("total_tokens", 0),
        )
        return 0

    except Exception as e:
        logger.error("cron_failed", run_id=run_id, error=str(e))
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

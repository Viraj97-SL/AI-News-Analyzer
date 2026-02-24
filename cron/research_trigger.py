import asyncio
import sys
import uuid

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("research_cron")
settings = get_settings()

async def main() -> int:
    run_id = str(uuid.uuid4())
    logger.info("research_cron_triggered", run_id=run_id, schedule="Thursday Deep Tech")

    try:
        # IMPORT THE NEW GRAPH HERE
        from app.agents.research_graph import build_research_graph

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

        if settings.is_sqlite:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
            graph = build_research_graph(checkpointer=checkpointer)
            result = await graph.ainvoke(initial_state, config)
        else:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            async with AsyncPostgresSaver.from_conn_string(settings.langgraph_pg_uri) as checkpointer:
                await checkpointer.setup()
                graph = build_research_graph(checkpointer=checkpointer)
                result = await graph.ainvoke(initial_state, config)

        logger.info(
            "research_cron_completed",
            run_id=run_id,
            status=result.get("approval_status"),
        )
        return 0

    except Exception as e:
        logger.error("research_cron_failed", run_id=run_id, error=str(e))
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
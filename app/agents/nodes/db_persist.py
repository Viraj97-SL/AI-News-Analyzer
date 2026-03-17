"""
Database persistence node — writes run data, articles, and summaries to
the application database (news_articles, summaries, agent_runs tables).

Runs after summarize_node so all enriched data is available.
Uses a synchronous SQLAlchemy session (psycopg2/sqlite) because pipeline
nodes are sync functions. Tables are created automatically if missing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_sync_session():
    """Return a synchronous SQLAlchemy Session using the psycopg2/sqlite driver."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.sync_database_url, pool_pre_ping=True)

    # Create all app tables if they don't exist (idempotent — safe to call every run)
    from app.models.models import Base
    Base.metadata.create_all(engine)

    return Session(engine)


def persist_to_db_node(state: PipelineState) -> dict:
    """
    Persist the current run's data to the application database.

    Writes:
      - AgentRun row (upsert — idempotent on revision loops)
      - NewsArticleModel rows for every deduplicated article
      - SummaryModel rows for every generated summary
    """
    from app.models.models import AgentRun, NewsArticleModel, RunStatus, SummaryModel

    run_id = state["run_id"]
    articles = state.get("deduplicated_articles", [])
    summaries = state.get("summaries", [])

    try:
        with _get_sync_session() as session:
            # ── AgentRun (upsert) ────────────────────────────────────────────
            existing_run = session.get(AgentRun, run_id)
            if not existing_run:
                session.add(
                    AgentRun(
                        id=run_id,
                        status=RunStatus.RUNNING,
                        trigger_type=state.get("trigger_type", "scheduled"),
                        started_at=datetime.now(UTC),
                    )
                )
            else:
                existing_run.status = RunStatus.RUNNING

            # ── NewsArticleModel rows ────────────────────────────────────────
            article_count = 0
            for article in articles:
                url = article.get("url", "")
                article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url or run_id))
                if session.get(NewsArticleModel, article_id) is None:
                    session.add(
                        NewsArticleModel(
                            id=article_id,
                            run_id=run_id,
                            title=article.get("title", "")[:500],
                            url=url[:2000],
                            source=article.get("source", "")[:100],
                            content=article.get("content", ""),
                            published_at=article.get("published_at"),
                            credibility_score=article.get("credibility_score", 0.0),
                            category=article.get("category", ""),
                        )
                    )
                    article_count += 1

            # ── SummaryModel rows ────────────────────────────────────────────
            summary_count = 0
            for i, summary in enumerate(summaries):
                summary_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:summary:{i}"))
                if session.get(SummaryModel, summary_id) is None:
                    source_urls_json = __import__("json").dumps(summary.get("source_urls", []))
                    session.add(
                        SummaryModel(
                            id=summary_id,
                            run_id=run_id,
                            headline=summary.get("headline", "")[:200],
                            body=summary.get("body", ""),
                            category=summary.get("category", "Other")[:50],
                            credibility_score=summary.get("credibility_score", 0.0),
                            source_urls=source_urls_json,
                        )
                    )
                    summary_count += 1

            session.commit()

        logger.info(
            "db_persisted",
            run_id=run_id,
            articles_saved=article_count,
            summaries_saved=summary_count,
        )

    except Exception as e:
        # DB persistence is non-critical — log and continue the pipeline
        logger.error("db_persist_failed", run_id=run_id, error=str(e))

    return {"current_step": "persisted"}


def persist_publish_result(run_id: str, linkedin_post_id: str | None, post_type: str) -> None:
    """
    Called from _publish_node to record the published LinkedIn post and mark
    the AgentRun as completed. Non-critical — errors are logged and swallowed.
    """
    from datetime import UTC, datetime

    from app.models.models import AgentRun, LinkedInPostModel, RunStatus

    try:
        with _get_sync_session() as session:
            # Mark run completed
            run = session.get(AgentRun, run_id)
            if run:
                run.status = RunStatus.COMPLETED
                run.completed_at = datetime.now(UTC)

            # Record LinkedIn post
            if linkedin_post_id:
                session.add(
                    LinkedInPostModel(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        content=f"[{post_type}]",
                        approval_status=__import__(
                            "app.models.models", fromlist=["ApprovalStatus"]
                        ).ApprovalStatus.APPROVED,
                        published_at=datetime.now(UTC),
                    )
                )

            session.commit()
        logger.info("publish_result_persisted", run_id=run_id, post_id=linkedin_post_id)
    except Exception as e:
        logger.error("persist_publish_result_failed", run_id=run_id, error=str(e))

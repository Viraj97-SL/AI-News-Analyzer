"""
Lightweight digest entry — for papers that clear ingestion but fail the
relevance gate (`relevance.py`). Rather than discarding the analysis work
entirely, produces a short one-paragraph write-up suitable for a "a few
papers worth a mention this week" style digest.

Batching design: papers are accumulated across separate pipeline runs and
combined into a single "N papers this week" email once enough have built
up — see `_persist_and_maybe_publish_digest_node` below for the mechanics
and the reasoning for checking the batch at the end of every run rather
than on a scheduler.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

if TYPE_CHECKING:
    from app.models.models import DigestEntryModel
    from app.agents.state import PipelineState

from app.agents.nodes.text_utils import normalize_text
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def build_digest_entry_node(state: "PipelineState") -> dict:
    paper = state.get("chosen_research_paper", {})
    analysis = state.get("deep_analysis", {})
    if not paper or not analysis:
        return {"digest_html": "", "current_step": "digest_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=settings.google_api_key,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Write ONE tight paragraph (max 70 words) summarizing this paper for a "
             "'a few papers worth knowing about this week' digest. State what it does "
             "and the single most concrete result. No hype words. Third person only — "
             "'the authors', never 'we'."),
            ("user", "Title: {title}\n\nCore problem: {core_problem}\n\nBreakthroughs: {breakthroughs}"),
        ])
        resp = (prompt | llm).invoke({
            "title": paper.get("title", ""),
            "core_problem": analysis.get("core_problem", ""),
            "breakthroughs": analysis.get("breakthroughs", ""),
        }).content
        paragraph = (
            "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in resp).strip()
            if isinstance(resp, list) else resp.strip()
        )
    except Exception as e:
        logger.warning("digest_entry_generation_failed", error=str(e))
        paragraph = analysis.get("core_problem", "")[:300]

    paragraph = normalize_text(paragraph)
    title = normalize_text(paper.get("title", ""))
    url = paper.get("url", "")
    relevance_score = state.get("relevance_score", 0.0)

    html = (
        '<div style="font-family:-apple-system,sans-serif;max-width:600px;margin:0 auto">'
        '<div style="font-family:monospace;font-size:11px;color:#64748B;letter-spacing:1px;'
        'text-transform:uppercase;margin-bottom:12px">Paper Digest — Quick Mention</div>'
        f'<h3 style="font-size:17px;margin:0 0 8px;color:#0F172A">{title}</h3>'
        f'<p style="font-size:14px;color:#334155;line-height:1.6;margin:0 0 10px">{paragraph}</p>'
        f'<a href="{url}" style="font-size:12px;color:#0EA5E9">{url}</a>'
        "</div>"
    )

    logger.info("digest_entry_built", title=title[:80], relevance_score=relevance_score)
    return {"digest_html": html, "current_step": "digest_built"}


def _build_combined_digest_html(entries: list["DigestEntryModel"]) -> str:
    """Compose the single "N papers this week" email body from N per-paper snippets."""
    items_html = "".join(entry.html_snippet for entry in entries)
    header = (
        '<div style="font-family:-apple-system,sans-serif;max-width:640px;margin:0 auto">'
        '<div style="font-family:monospace;font-size:12px;color:#334155;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:20px">{len(entries)} Papers Worth Knowing '
        'About This Week</div>'
    )
    return f"{header}{items_html}</div>"


def _persist_and_maybe_publish_digest_node(state: "PipelineState") -> dict:
    """
    Persist this run's digest entry, then publish one combined "N papers this
    week" email once ``settings.digest_batch_size`` unpublished entries have
    accumulated across runs — otherwise just wait for more.

    Design choice — batch readiness is checked at the END of every run,
    not on a separate scheduler: pipeline runs are triggered externally
    (see `app/api/v1/routes/runs.py::trigger_run`, a plain FastAPI
    BackgroundTask with no APScheduler/cron loop anywhere in `app/main.py`).
    Because there is no long-lived process to host a periodic job, the only
    reliable moment to notice "we now have enough papers" is right after
    the current run's entry is written. This needs no new scheduling
    infrastructure and fails safe: if the batch isn't full yet, the row is
    simply left `published=False` and picked up by whichever run pushes the
    count over the threshold next.
    """
    from app.agents.nodes.db_persist import _get_sync_session
    from app.models.models import DigestEntryModel

    run_id = state.get("run_id", "")
    digest_html = state.get("digest_html", "")
    if not digest_html:
        return {"current_step": "digest_publish_skipped"}

    paper = state.get("chosen_research_paper", {})
    title = normalize_text(paper.get("title", ""))[:500]
    url = paper.get("url", "")[:2000]
    relevance_score = state.get("relevance_score", 0.0)
    entry_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:digest_entry"))

    try:
        with _get_sync_session() as session:
            if session.get(DigestEntryModel, entry_id) is None:
                session.add(
                    DigestEntryModel(
                        id=entry_id,
                        run_id=run_id,
                        title=title,
                        url=url,
                        html_snippet=digest_html,
                        relevance_score=relevance_score,
                        published=False,
                    )
                )
                session.commit()

            unpublished = (
                session.query(DigestEntryModel)
                .filter(DigestEntryModel.published.is_(False))
                .order_by(DigestEntryModel.created_at.asc())
                .all()
            )

            if len(unpublished) < settings.digest_batch_size:
                logger.info(
                    "digest_entry_queued",
                    run_id=run_id,
                    queued=len(unpublished),
                    batch_size=settings.digest_batch_size,
                )
                return {"current_step": "digest_queued"}

            from app.services.email_service import EmailService

            try:
                EmailService().send_newsletter(
                    html_content=_build_combined_digest_html(unpublished),
                    subject=f"🗞️ AI Research Digest — {len(unpublished)} papers this week",
                )
                for entry in unpublished:
                    entry.published = True
                session.commit()
                logger.info("digest_batch_published", run_id=run_id, count=len(unpublished))
            except Exception as e:
                logger.error("digest_batch_publish_failed", run_id=run_id, error=str(e))

    except Exception as e:
        logger.error("digest_persist_failed", run_id=run_id, error=str(e))
        return {"current_step": "digest_publish_skipped"}

    return {"current_step": "digest_published"}

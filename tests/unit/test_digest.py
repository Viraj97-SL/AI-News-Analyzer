"""Tests for digest.py — the lightweight fallback content type for papers
that fail the relevance gate, and its cross-run batching/publish logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agents.nodes.digest import (
    _persist_and_maybe_publish_digest_node,
    build_digest_entry_node,
)
from app.models.models import Base, DigestEntryModel


class TestBuildDigestEntryNode:
    def test_skips_when_no_paper_or_analysis(self):
        result = build_digest_entry_node({"chosen_research_paper": {}, "deep_analysis": {}})
        assert result["digest_html"] == ""
        assert result["current_step"] == "digest_skipped"

    def test_builds_html_with_title_and_paragraph(self):
        mock_response = MagicMock()
        mock_response.content = "A short paragraph about the paper."
        mock_llm = MagicMock()
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response

        with patch("app.agents.nodes.digest.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = mock_llm
            result = build_digest_entry_node({
                "chosen_research_paper": {"title": "A Paper Title", "url": "https://arxiv.org/abs/1"},
                "deep_analysis": {"core_problem": "A problem.", "breakthroughs": "A result."},
                "relevance_score": 3.2,
            })

        assert "A Paper Title" in result["digest_html"]
        assert "A short paragraph about the paper." in result["digest_html"]
        assert result["current_step"] == "digest_built"

    def test_falls_back_to_core_problem_on_llm_failure(self):
        with patch("app.agents.nodes.digest.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = build_digest_entry_node({
                "chosen_research_paper": {"title": "A Paper Title", "url": "https://arxiv.org/abs/1"},
                "deep_analysis": {"core_problem": "The fallback problem text.", "breakthroughs": ""},
            })

        assert "The fallback problem text" in result["digest_html"]
        assert result["current_step"] == "digest_built"

    def test_title_is_latex_normalized(self):
        with patch("app.agents.nodes.digest.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("skip llm")
            result = build_digest_entry_node({
                "chosen_research_paper": {"title": "M$^\\text{4}$World", "url": "https://arxiv.org/abs/1"},
                "deep_analysis": {"core_problem": "text", "breakthroughs": ""},
            })

        assert "M⁴World" in result["digest_html"]
        assert "$" not in result["digest_html"].split("<p")[0]


@pytest.fixture
def in_memory_session_factory():
    """
    A shared in-memory SQLite DB (via StaticPool, so every `_get_sync_session()`
    call reuses the same connection instead of getting a fresh empty database)
    standing in for the real DB layer — lets the batching logic run against
    real SQLAlchemy query/filter/commit behavior without touching the app's
    configured database.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    def _factory():
        return Session(engine)

    return _factory


def _make_state(run_id: str, title: str = "Paper", relevance_score: float = 3.0) -> dict:
    return {
        "run_id": run_id,
        "digest_html": f'<div><h3>{title}</h3><p>paragraph</p></div>',
        "chosen_research_paper": {"title": title, "url": f"https://arxiv.org/abs/{run_id}"},
        "relevance_score": relevance_score,
    }


class TestPersistAndMaybePublishDigestNode:
    def test_skips_when_no_digest_html(self, in_memory_session_factory):
        with patch("app.agents.nodes.db_persist._get_sync_session", in_memory_session_factory):
            result = _persist_and_maybe_publish_digest_node({"digest_html": ""})
        assert result["current_step"] == "digest_publish_skipped"

    def test_paper_added_but_batch_not_full_sends_no_email(self, in_memory_session_factory):
        with patch("app.agents.nodes.db_persist._get_sync_session", in_memory_session_factory), \
             patch("app.agents.nodes.digest.settings.digest_batch_size", 3), \
             patch("app.services.email_service.EmailService") as MockEmailService:
            result = _persist_and_maybe_publish_digest_node(_make_state("run-1"))

        MockEmailService.assert_not_called()
        assert result["current_step"] == "digest_queued"

        with in_memory_session_factory() as session:
            rows = session.query(DigestEntryModel).all()
        assert len(rows) == 1
        assert rows[0].published is False

    def test_paper_completing_batch_sends_one_combined_email_and_marks_published(
        self, in_memory_session_factory
    ):
        mock_instance = MagicMock()

        with patch("app.agents.nodes.db_persist._get_sync_session", in_memory_session_factory), \
             patch("app.agents.nodes.digest.settings.digest_batch_size", 3):
            # First two papers just queue.
            _persist_and_maybe_publish_digest_node(_make_state("run-1", "Paper One"))
            _persist_and_maybe_publish_digest_node(_make_state("run-2", "Paper Two"))

            with patch("app.services.email_service.EmailService") as MockEmailService:
                MockEmailService.return_value = mock_instance
                result = _persist_and_maybe_publish_digest_node(_make_state("run-3", "Paper Three"))

        mock_instance.send_newsletter.assert_called_once()
        call_kwargs = mock_instance.send_newsletter.call_args.kwargs
        assert "Paper One" in call_kwargs["html_content"]
        assert "Paper Two" in call_kwargs["html_content"]
        assert "Paper Three" in call_kwargs["html_content"]
        assert "3" in call_kwargs["subject"]
        assert result["current_step"] == "digest_published"

        with in_memory_session_factory() as session:
            rows = session.query(DigestEntryModel).all()
        assert len(rows) == 3
        assert all(row.published for row in rows)

    def test_email_failure_does_not_raise_and_leaves_entries_unpublished(
        self, in_memory_session_factory
    ):
        with patch("app.agents.nodes.db_persist._get_sync_session", in_memory_session_factory), \
             patch("app.agents.nodes.digest.settings.digest_batch_size", 1), \
             patch("app.services.email_service.EmailService") as MockEmailService:
            MockEmailService.return_value.send_newsletter.side_effect = RuntimeError("SMTP down")
            result = _persist_and_maybe_publish_digest_node(_make_state("run-1"))

        assert result["current_step"] == "digest_published"

        with in_memory_session_factory() as session:
            rows = session.query(DigestEntryModel).all()
        assert len(rows) == 1
        assert rows[0].published is False

    def test_running_same_run_id_twice_does_not_duplicate_entry(self, in_memory_session_factory):
        with patch("app.agents.nodes.db_persist._get_sync_session", in_memory_session_factory), \
             patch("app.agents.nodes.digest.settings.digest_batch_size", 3), \
             patch("app.services.email_service.EmailService"):
            _persist_and_maybe_publish_digest_node(_make_state("run-1"))
            _persist_and_maybe_publish_digest_node(_make_state("run-1"))

        with in_memory_session_factory() as session:
            rows = session.query(DigestEntryModel).all()
        assert len(rows) == 1

"""
Shared pytest fixtures for unit and integration tests.

Uses FakeListChatModel for deterministic LLM mocking — no API keys needed.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents.state import NewsArticle, PipelineState


@pytest.fixture
def mock_llm() -> FakeListChatModel:
    """Deterministic mock LLM that returns canned responses."""
    return FakeListChatModel(
        responses=[
            '[{"headline": "Test AI News", "body": "A test summary.", "category": "LLM",'
            ' "source_url": "https://example.com", "credibility_score": 0.8}]',
        ]
    )


@pytest.fixture
def sample_article() -> NewsArticle:
    """A realistic sample article for testing."""
    return NewsArticle(
        title="OpenAI Releases GPT-5 with Reasoning Capabilities",
        url="https://techcrunch.com/2025/01/15/openai-gpt5",
        source="rss:techcrunch",
        content=(
            "OpenAI has announced GPT-5, its latest large language model featuring "
            "advanced reasoning capabilities. The model demonstrates significant "
            "improvements in mathematical problem-solving, code generation, and "
            "multi-step logical reasoning. Early benchmarks show a 40% improvement "
            "over GPT-4o on complex reasoning tasks."
        ),
        published_at="2025-01-15T10:00:00Z",
        credibility_score=0.0,
    )


@pytest.fixture
def sample_articles(sample_article: NewsArticle) -> list[NewsArticle]:
    """Multiple sample articles for pipeline testing."""
    return [
        sample_article,
        NewsArticle(
            title="Google DeepMind Achieves Protein Folding Breakthrough",
            url="https://www.nature.com/articles/deepmind-protein",
            source="rss:nature",
            content="DeepMind's AlphaFold 3 can now predict protein interactions...",
            published_at="2025-01-14T08:00:00Z",
            credibility_score=0.0,
        ),
        NewsArticle(
            title="EU AI Act Enforcement Begins",
            url="https://reuters.com/eu-ai-act",
            source="tavily",
            content="The European Union's AI Act officially entered enforcement...",
            published_at="2025-01-13T14:00:00Z",
            credibility_score=0.0,
        ),
    ]


@pytest.fixture
def initial_state(sample_articles: list[NewsArticle]) -> PipelineState:
    """A fully populated initial pipeline state for testing."""
    return PipelineState(
        run_id="test-run-001",
        trigger_type="manual",
        raw_articles=sample_articles,
        deduplicated_articles=[],
        summaries=[],
        newsletter_html="",
        linkedin_draft="",
        image_paths=[],
        approval_status="pending",
        feedback="",
        error_log=[],
        total_tokens=0,
        total_cost=0.0,
        current_step="starting",
    )

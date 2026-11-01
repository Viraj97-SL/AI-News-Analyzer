"""Graph wiring + prompt-context tests for research pipeline v2."""

from __future__ import annotations

from app.agents.research_graph import (
    RichDeepAnalysis,
    build_research_graph,
    paper_context_block,
    select_paper_node,
)


def test_research_graph_uses_research_only_sources():
    nodes = set(build_research_graph().get_graph().nodes)
    assert {"scrape_research_arxiv", "scrape_hf_daily", "load_manual_papers"} <= nodes
    assert "scrape_arxiv" not in nodes  # the shared news scraper stays out of the research graph


def test_select_paper_prefers_enriched_candidates():
    raw = {"title": "Raw", "url": "https://arxiv.org/abs/2610.1", "content": "c"}
    enriched = {**raw, "lab": "Anthropic", "trend_tags": ["rag"]}
    out = select_paper_node(
        {
            "raw_articles": [raw],
            "research_candidates": [enriched],
            "paper_rankings": [{"paper_url": raw["url"], "composite_score": 9}],
        }
    )
    assert out["chosen_research_paper"]["lab"] == "Anthropic"
    assert out["chosen_research_paper"]["trend_tags"] == ["rag"]


def test_paper_context_block_with_lab_and_trends():
    block = paper_context_block({"lab": "Google DeepMind", "trend_tags": ["rag", "multi_agent"]})
    assert "Google DeepMind researchers" in block
    assert "RAG, Multi-Agent Systems" in block
    assert "#RAG #MultiAgentSystems" in block


def test_paper_context_block_without_signals():
    block = paper_context_block({})
    assert "unknown" in block
    assert "TREND CONTEXT: none" in block


def test_rich_analysis_why_now_is_optional():
    fields = RichDeepAnalysis.model_fields
    assert "why_now" in fields
    assert fields["why_now"].default == ""

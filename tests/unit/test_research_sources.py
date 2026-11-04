"""Tests for research-only paper sourcing (HF Daily Papers, expanded arXiv, merge/enrich)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.agents.nodes import research_sources as rs

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _hf_item(
    pid="2610.11111", days_old=1, upvotes=40, org="Google", stars=120, title="Agentic RAG at scale"
):
    ts = (NOW - timedelta(days=days_old)).isoformat().replace("+00:00", "Z")
    return {
        "publishedAt": ts,
        "title": title,
        "organization": {"name": org.lower(), "fullname": org} if org else None,
        "paper": {
            "id": pid,
            "title": title,
            "summary": "We study retrieval-augmented generation with multi-agent planners.",
            "upvotes": upvotes,
            "publishedAt": ts,
            "githubRepo": "https://github.com/x/y",
            "githubStars": stars,
            "ai_keywords": ["MCP", "tool calling"],
            "authors": [{"name": "A. Author"}],
        },
    }


# ── canonical ids ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://arxiv.org/abs/2610.11111v2", "2610.11111"),
        ("https://arxiv.org/abs/2610.11111", "2610.11111"),
        ("https://arxiv.org/pdf/2610.11111v1.pdf", "2610.11111"),
        ("https://huggingface.co/papers/2610.11111", "2610.11111"),
        ("https://example.com/blog", None),
    ],
)
def test_canonical_arxiv_id(url, expected):
    assert rs.canonical_arxiv_id(url) == expected


# ── HF Daily Papers ──────────────────────────────────────────────────────────


def test_parse_hf_daily_papers_maps_fields():
    articles = rs.parse_hf_daily_papers([_hf_item()], now=NOW)
    assert len(articles) == 1
    a = articles[0]
    assert a["url"] == "https://arxiv.org/abs/2610.11111"
    assert a["source"] == "hf_daily"
    assert a["upvotes"] == 40
    assert a["github_stars"] == 120
    assert a["affiliation_hint"] == "Google"
    assert "MCP" in a["content"]  # ai_keywords appended to aid trend tagging


def test_parse_hf_daily_papers_drops_old_and_malformed():
    items = [_hf_item(days_old=30), {"paper": {}}, {"nope": 1}, _hf_item(pid="2610.22222")]
    articles = rs.parse_hf_daily_papers(items, now=NOW)
    assert [a["url"] for a in articles] == ["https://arxiv.org/abs/2610.22222"]


def test_parse_hf_daily_papers_handles_missing_org():
    a = rs.parse_hf_daily_papers([_hf_item(org=None)], now=NOW)[0]
    assert a["affiliation_hint"] == ""


def test_scrape_hf_daily_papers_node_success():
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = [_hf_item()]
    response.raise_for_status.return_value = None
    client = MagicMock()
    client.__enter__.return_value.get.return_value = response
    with (
        patch.object(rs.httpx, "Client", return_value=client),
        patch.object(rs, "_now", return_value=NOW),
    ):
        out = rs.scrape_hf_daily_papers_node({})
    assert len(out["raw_articles"]) == 1


def test_scrape_hf_daily_papers_node_failure_is_non_fatal():
    client = MagicMock()
    client.__enter__.return_value.get.side_effect = httpx.ConnectError("down")
    with patch.object(rs.httpx, "Client", return_value=client):
        out = rs.scrape_hf_daily_papers_node({})
    assert out["raw_articles"] == []
    assert "HF Daily Papers" in out["error_log"][0]


# ── arXiv ────────────────────────────────────────────────────────────────────


def _arxiv_result(
    entry="http://arxiv.org/abs/2610.33333v1", days_old=1, comment="Work done at Meta FAIR"
):
    return SimpleNamespace(
        title="A Paper",
        entry_id=entry,
        summary="Abstract about agentic systems.",
        published=NOW - timedelta(days=days_old),
        comment=comment,
        journal_ref=None,
    )


def test_research_arxiv_query_includes_agent_and_ir_categories():
    assert "cat:cs.MA" in rs.research_arxiv_query()
    assert "cat:cs.IR" in rs.research_arxiv_query()


def test_scrape_research_arxiv_node_filters_by_date_and_keeps_comment_hint():
    fake_client = MagicMock()
    fake_client.results.return_value = [
        _arxiv_result(),
        _arxiv_result(entry="http://arxiv.org/abs/old", days_old=20),
    ]
    with (
        patch("arxiv.Client", return_value=fake_client),
        patch.object(rs, "_now", return_value=NOW),
    ):
        out = rs.scrape_research_arxiv_node({})
    arts = out["raw_articles"]
    assert len(arts) == 1
    assert arts[0]["source"] == "arxiv"
    assert arts[0]["affiliation_hint"] == "Work done at Meta FAIR"


def test_scrape_research_arxiv_node_failure_is_non_fatal():
    with patch("arxiv.Client", side_effect=RuntimeError("boom")):
        out = rs.scrape_research_arxiv_node({})
    assert out["raw_articles"] == []
    assert out["error_log"]


# ── merge & enrich ───────────────────────────────────────────────────────────


def test_merge_dedupes_by_arxiv_id_and_merges_signals():
    arxiv_copy = {
        "title": "T",
        "url": "http://arxiv.org/abs/2610.11111v1",
        "source": "arxiv",
        "content": "abstract",
        "published_at": "x",
        "credibility_score": 0.8,
        "affiliation_hint": "",
    }
    hf_copy = rs.parse_hf_daily_papers([_hf_item()], now=NOW)[0]
    merged = rs.merge_research_candidates([arxiv_copy, hf_copy])
    assert len(merged) == 1
    m = merged[0]
    assert m["upvotes"] == 40
    assert m["lab"] == "Google DeepMind"
    assert m["lab_tier"] == 1
    assert {"rag", "multi_agent", "mcp_tool_use"} <= set(m["trend_tags"])
    assert set(m["sources"]) == {"arxiv", "hf_daily"}


def test_merge_keeps_manual_source_priority():
    manual = {
        "title": "T",
        "url": "https://arxiv.org/abs/2610.11111",
        "source": "manual",
        "content": "c",
        "published_at": "x",
        "credibility_score": 1.0,
    }
    hf_copy = rs.parse_hf_daily_papers([_hf_item()], now=NOW)[0]
    merged = rs.merge_research_candidates([hf_copy, manual])
    assert merged[0]["source"] == "manual"
    assert merged[0]["url"] == "https://arxiv.org/abs/2610.11111"


def test_merge_keeps_non_arxiv_items_separately():
    a = {
        "title": "A",
        "url": "https://example.com/a",
        "source": "manual",
        "content": "",
        "published_at": "",
        "credibility_score": 1.0,
    }
    b = {
        "title": "B",
        "url": "https://example.com/b",
        "source": "manual",
        "content": "",
        "published_at": "",
        "credibility_score": 1.0,
    }
    assert len(rs.merge_research_candidates([a, b])) == 2


def test_merge_does_not_mutate_inputs():
    hf_copy = rs.parse_hf_daily_papers([_hf_item()], now=NOW)[0]
    snapshot = dict(hf_copy)
    rs.merge_research_candidates([hf_copy])
    assert hf_copy == snapshot


def test_attention_bonus_scales_and_caps():
    assert rs.attention_bonus({}) == 0.0
    low = rs.attention_bonus({"upvotes": 3})
    high = rs.attention_bonus({"upvotes": 120, "github_stars": 5000})
    assert 0 < low < high <= rs.MAX_ATTENTION_BONUS


# ── review fixes ─────────────────────────────────────────────────────────────


def test_canonical_arxiv_id_ignores_non_arxiv_urls_with_digits():
    assert rs.canonical_arxiv_id("https://example.com/report-2024.12345") is None


def test_research_arxiv_query_is_trend_focused():
    q = rs.research_arxiv_query()
    assert ") AND (" in q
    assert 'abs:"model context protocol"' in q


def test_merge_skips_articles_without_url_and_tolerates_none_content():
    good = {
        "title": "A",
        "url": "https://arxiv.org/abs/2610.00001",
        "source": "arxiv",
        "content": None,
    }
    no_url = {"title": "B", "url": "", "source": "manual", "content": "x"}
    merged = rs.merge_research_candidates([good, no_url])
    assert len(merged) == 1
    assert merged[0]["content"] == ""


def test_merge_uses_strict_matching_for_arxiv_comments():
    art = {
        "title": "T",
        "url": "https://arxiv.org/abs/2610.00002",
        "source": "arxiv",
        "content": "c",
        "affiliation_hint": "Our agent is built on Qwen",
    }
    assert rs.merge_research_candidates([art])[0]["lab"] is None

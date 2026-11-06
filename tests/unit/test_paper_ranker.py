"""Tests for rank_papers_node with superlab / attention / trend boosts (v2 phases 2-3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agents.nodes import paper_ranker as pr


def _article(pid, source="arxiv", hint="", upvotes=0, content="plain abstract", title=None):
    return {
        "title": title or f"Paper {pid}",
        "url": f"https://arxiv.org/abs/{pid}",
        "source": source,
        "content": content,
        "published_at": "2026-10-30T00:00:00Z",
        "credibility_score": 0.8,
        "affiliation_hint": hint,
        "upvotes": upvotes,
    }


def _uniform_llm(score=6):
    """Patch target that makes every paper get the same LLM quality scores."""

    def fake_invoke(inputs):
        urls = [
            line[5:].strip() for line in inputs["papers"].splitlines() if line.startswith("URL: ")
        ]
        return pr._BatchRanking(
            rankings=[
                pr._PaperScore(
                    paper_url=u,
                    novelty=score,
                    impact=score,
                    technical_depth=score,
                    benchmark_quality=score,
                    reproducibility=score,
                    reason="r",
                )
                for u in urls
            ]
        )

    chain = MagicMock()
    chain.invoke.side_effect = fake_invoke
    return chain


@pytest.fixture
def uniform_llm():
    with patch.object(pr, "_build_ranking_chain", return_value=_uniform_llm()):
        yield


def test_superlab_paper_outranks_equal_quality_unknown_lab(uniform_llm):
    arts = [_article("2610.00001"), _article("2610.00002", source="hf_daily", hint="OpenAI")]
    out = pr.rank_papers_node({"raw_articles": arts})
    top = out["paper_rankings"][0]
    assert top["paper_url"].endswith("2610.00002")
    assert top["lab"] == "OpenAI"
    assert top["lab_bonus"] == pytest.approx(1.5)


def test_manual_paper_still_beats_superlab(uniform_llm):
    arts = [
        _article(
            "2610.00002",
            source="hf_daily",
            hint="Google DeepMind",
            upvotes=200,
            content="multi-agent RAG with MCP tool calling",
        ),
        _article("2610.00003", source="manual"),
    ]
    out = pr.rank_papers_node({"raw_articles": arts})
    assert out["paper_rankings"][0]["is_manual"] is True


def test_trend_and_attention_bonuses_recorded(uniform_llm):
    arts = [
        _article(
            "2610.00004",
            source="hf_daily",
            upvotes=63,
            content="A retrieval-augmented multi-agent system",
        )
    ]
    top = pr.rank_papers_node({"raw_articles": arts})["paper_rankings"][0]
    assert set(top["trend_tags"]) >= {"rag", "multi_agent"}
    assert top["trend_bonus"] == pytest.approx(1.0)
    assert top["attention_bonus"] > 1.0
    assert top["composite_score"] == pytest.approx(
        top["quality_score"] + top["lab_bonus"] + top["attention_bonus"] + top["trend_bonus"],
        abs=0.01,
    )


def test_duplicates_across_sources_are_ranked_once(uniform_llm):
    arts = [
        _article("2610.00005"),
        _article("2610.00005", source="hf_daily", hint="Anthropic", upvotes=10),
    ]
    out = pr.rank_papers_node({"raw_articles": arts})
    assert len(out["paper_rankings"]) == 1
    assert len(out["research_candidates"]) == 1
    assert out["research_candidates"][0]["lab"] == "Anthropic"


def test_prefilter_caps_llm_workload_but_keeps_manual():
    arts = [_article(f"2610.{i:05d}") for i in range(pr.MAX_LLM_RANKED + 15)]
    arts.append(_article("2610.99999", source="manual"))
    chain = _uniform_llm()
    with patch.object(pr, "_build_ranking_chain", return_value=chain):
        out = pr.rank_papers_node({"raw_articles": arts})
    assert len(out["paper_rankings"]) == pr.MAX_LLM_RANKED
    assert any(r["is_manual"] for r in out["paper_rankings"])


def test_prefilter_prefers_signal_rich_papers():
    plain = [_article(f"2610.{i:05d}") for i in range(pr.MAX_LLM_RANKED)]
    hot = _article("2610.88888", source="hf_daily", hint="Meta", upvotes=100)
    selected = pr.select_for_llm_ranking(pr.merge_research_candidates([*plain, hot]))
    assert any(a["url"].endswith("2610.88888") for a in selected)


def test_llm_batch_failure_falls_back_to_neutral_scores():
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("quota")
    with patch.object(pr, "_build_ranking_chain", return_value=chain):
        out = pr.rank_papers_node(
            {"raw_articles": [_article("2610.00006", hint="NVIDIA", source="hf_daily")]}
        )
    top = out["paper_rankings"][0]
    assert top["quality_score"] == pytest.approx(5.0)
    assert top["lab"] == "NVIDIA"


def test_empty_articles():
    out = pr.rank_papers_node({"raw_articles": []})
    assert out["paper_rankings"] == []


# ── AI-topic filter ────────────────────────────────────────────────────────
# Regression: "A Quasipolynomial Algorithm for ell_infty Contractions" (pure math)
# was picked as a run's paper and reached the 2026-11-06 digest email.
_OFF_TOPIC_URLS = {"https://arxiv.org/abs/2610.30122"}


def _topic_aware_llm():
    def fake_invoke(inputs):
        urls = [
            line[5:].strip() for line in inputs["papers"].splitlines() if line.startswith("URL: ")
        ]
        return pr._BatchRanking(
            rankings=[
                pr._PaperScore(
                    paper_url=u,
                    novelty=9 if u in _OFF_TOPIC_URLS else 6,  # off-topic paper scores HIGHER
                    impact=9 if u in _OFF_TOPIC_URLS else 6,
                    technical_depth=9 if u in _OFF_TOPIC_URLS else 6,
                    benchmark_quality=6,
                    reproducibility=6,
                    reason="r",
                    is_ai_ml=u not in _OFF_TOPIC_URLS,
                )
                for u in urls
            ]
        )

    chain = MagicMock()
    chain.invoke.side_effect = fake_invoke
    return chain


def test_off_topic_paper_is_dropped_even_if_it_scores_highest():
    arts = [_article("2610.30122", title="A Quasipolynomial Algorithm for ell_infty Contractions"),
            _article("2610.00001")]
    with patch.object(pr, "_build_ranking_chain", return_value=_topic_aware_llm()):
        out = pr.rank_papers_node({"raw_articles": arts})

    urls = [r["paper_url"] for r in out["paper_rankings"]]
    assert urls == ["https://arxiv.org/abs/2610.00001"]


def test_manual_paper_is_never_dropped_as_off_topic():
    arts = [_article("2610.30122", source="manual"), _article("2610.00001")]
    with patch.object(pr, "_build_ranking_chain", return_value=_topic_aware_llm()):
        out = pr.rank_papers_node({"raw_articles": arts})

    assert out["paper_rankings"][0]["paper_url"].endswith("2610.30122")


def test_ranking_failure_keeps_papers_fail_open():
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("quota")
    with patch.object(pr, "_build_ranking_chain", return_value=chain):
        out = pr.rank_papers_node({"raw_articles": [_article("2610.30122")]})

    assert len(out["paper_rankings"]) == 1


def test_is_ai_ml_defaults_true_when_model_omits_it():
    score = pr._PaperScore(
        paper_url="u", novelty=5, impact=5, technical_depth=5,
        benchmark_quality=5, reproducibility=5, reason="r",
    )
    assert score.is_ai_ml is True

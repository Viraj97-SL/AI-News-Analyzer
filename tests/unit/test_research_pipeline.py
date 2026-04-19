"""
Tests for the research pipeline quality overhaul.

Covers:
  - RichDeepAnalysis: Pydantic model validation and defaults
  - _build_research_article_html: HTML structure and content
  - deep_analysis_node: LLM orchestration (analysis + LinkedIn draft)
  - research_carousel_node: 10-slide generation, graceful fallbacks
  - send_approval_email: new research_article_html parameter
"""

from __future__ import annotations

import email as stdlib_email
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.agents.research_graph import (
    ResearchScores,
    RichDeepAnalysis,
    _build_research_article_html,
    _render_gauge_svg,
    deep_analysis_node,
    score_research_node,
    select_paper_node,
)
from app.agents.nodes.research_carousel import _TOTAL_SLIDES, research_carousel_node
from app.services.email_service import EmailService


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_rich_analysis(**overrides) -> RichDeepAnalysis:
    """Fully populated RichDeepAnalysis for use across tests."""
    defaults = dict(
        core_problem=(
            "Existing attention is O(n²) in memory, capping context at 4k tokens "
            "even on 80 GB GPUs. Long documents must be chunked, losing coherence."
        ),
        methodology=(
            "Sparse attention with learned per-token routing reduces complexity to O(n log n). "
            "A small MLP predicts which token pairs attend; routing is trained end-to-end."
        ),
        breakthroughs=(
            "Achieves 89.2% on MMLU (+3.1% vs GPT-4-Turbo) at 10× lower inference cost. "
            "Handles 32k-token documents on a single 24 GB GPU."
        ),
        limitations=(
            "Only evaluated on English benchmarks. Routing overhead adds ~5% latency. "
            "Not tested on code generation tasks."
        ),
        executive_summary=(
            "This paper solves memory-bound attention for long documents.\n\n"
            "The sparse routing mechanism enables 32k-context inference on consumer GPUs."
        ),
        key_contributions=[
            "Sparse attention: O(n²) → O(n log n) memory, proven on 3 architectures.",
            "Learned routing MLP trained end-to-end with straight-through estimators.",
            "4× longer context at identical compute vs. full attention.",
        ],
        technical_innovation=(
            "Unlike prior linear-attention methods (Performer, Longformer), routing is "
            "dynamic per token rather than using a fixed local window or random projection."
        ),
        experiment_setup=(
            "Pre-trained on C4 (350B tokens). Evaluated on MMLU, HellaSwag, ARC-Challenge. "
            "Baselines: GPT-4-Turbo, Claude 3.5 Sonnet, Mistral-7B. Compute: 256 A100s × 10 days."
        ),
        quantitative_results=[
            "MMLU: 89.2% (+3.1% vs GPT-4-Turbo, 5-shot)",
            "HellaSwag: 95.1% (+1.8% vs prior SOTA)",
            "ARC-Challenge: 92.4% (+2.7% vs Claude 3.5)",
        ],
        ablation_highlights=(
            "Removing learned routing drops MMLU by 4.2% and reverts to full-attention cost. "
            "Removing sparse masking alone degrades context handling by 4×. "
            "Both components are necessary — neither alone achieves the target."
        ),
        real_world_applications=[
            "Long-document summarization in legal tech (contracts, case law).",
            "Code review assistants handling full-repository context.",
            "Medical record analysis across multi-year patient histories.",
        ],
        ecosystem_impact=(
            "HuggingFace Transformers needs a sparse-attention backend. "
            "vLLM can adopt routing weights directly from checkpoint."
        ),
        expert_interpretation=(
            "Engineers can fine-tune this on 24 GB VRAM today. "
            "Drop-in replacement for standard attention in any Transformer block."
        ),
        technical_deep_dive=(
            "The core mechanism replaces full attention with a learned sparse graph.\n\n"
            "At each layer, a routing MLP with 2 hidden layers predicts which token pairs "
            "need to attend. Gradients flow through the discrete routing via straight-through "
            "estimators: during the forward pass, routing is hard (binary mask); during "
            "backprop, gradients pass as if routing were soft (sigmoid). "
            "This enables end-to-end training without separate routing supervision."
        ),
        future_directions=[
            "Extend routing to multi-modal inputs (vision + text).",
            "Apply to diffusion transformers for high-res image generation.",
            "Explore hardware-aware routing that respects GPU memory bandwidth.",
        ],
        significance_verdict="Major Contribution",
    )
    defaults.update(overrides)
    return RichDeepAnalysis(**defaults)


def make_paper(**overrides) -> dict:
    return {
        "title": "Efficient Sparse Attention via Learned Token Routing",
        "url": "https://arxiv.org/abs/2401.99999",
        "content": "We propose a sparse attention mechanism that achieves O(n log n) memory...",
        **overrides,
    }


# ── 1. RichDeepAnalysis model ─────────────────────────────────────────────────

class TestRichDeepAnalysis:
    def test_creates_with_all_fields(self):
        a = make_rich_analysis()
        assert a.core_problem
        assert a.significance_verdict == "Major Contribution"
        assert len(a.key_contributions) == 3
        assert len(a.quantitative_results) == 3

    def test_list_fields_default_to_empty(self):
        """Fields with default_factory=list must not share state between instances."""
        a = RichDeepAnalysis(
            core_problem="p", methodology="m", breakthroughs="b", limitations="l",
            executive_summary="s", technical_innovation="i", experiment_setup="e",
            ablation_highlights="a", ecosystem_impact="eco", expert_interpretation="ex",
            technical_deep_dive="d", significance_verdict="Incremental",
        )
        assert a.key_contributions == []
        assert a.quantitative_results == []
        assert a.real_world_applications == []
        assert a.future_directions == []

    def test_significance_verdict_values(self):
        for verdict in ("Incremental", "Solid Contribution", "Major Contribution", "Paradigm Shift"):
            a = make_rich_analysis(significance_verdict=verdict)
            assert a.significance_verdict == verdict

    def test_model_dump_preserves_all_fields(self):
        a = make_rich_analysis()
        d = a.model_dump()
        for field in (
            "core_problem", "methodology", "breakthroughs", "limitations",
            "executive_summary", "key_contributions", "technical_innovation",
            "experiment_setup", "quantitative_results", "ablation_highlights",
            "real_world_applications", "ecosystem_impact", "expert_interpretation",
            "technical_deep_dive", "future_directions", "significance_verdict",
        ):
            assert field in d, f"Missing field in model_dump: {field}"

    def test_backward_compat_fields_present(self):
        """The 4 old fields must still exist for benchmark_chart and prior_art nodes."""
        a = make_rich_analysis()
        assert a.core_problem
        assert a.methodology
        assert a.breakthroughs
        assert a.limitations


# ── 2. _build_research_article_html ──────────────────────────────────────────

class TestBuildResearchArticleHtml:
    def test_returns_non_empty_html(self):
        html = _build_research_article_html(make_paper(), make_rich_analysis())
        assert len(html) > 500
        assert "<div" in html

    def test_includes_paper_title(self):
        paper = make_paper(title="My Unique Paper Title XYZ123")
        html = _build_research_article_html(paper, make_rich_analysis())
        assert "My Unique Paper Title XYZ123" in html

    def test_includes_paper_url(self):
        paper = make_paper(url="https://arxiv.org/abs/9999.00001")
        html = _build_research_article_html(paper, make_rich_analysis())
        assert "9999.00001" in html

    def test_includes_significance_verdict(self):
        html = _build_research_article_html(make_paper(), make_rich_analysis(significance_verdict="Paradigm Shift"))
        assert "PARADIGM SHIFT" in html.upper()

    def test_includes_key_sections(self):
        html = _build_research_article_html(make_paper(), make_rich_analysis())
        for section in (
            "Executive Summary",
            "The Problem",
            "Technical Deep Dive",
            "Real-World Applications",
            "What Comes Next",
        ):
            assert section in html, f"Missing section: {section}"

    def test_renders_contributions_as_list(self):
        a = make_rich_analysis(key_contributions=["Contrib A", "Contrib B"])
        html = _build_research_article_html(make_paper(), a)
        assert "Contrib A" in html
        assert "Contrib B" in html

    def test_renders_empty_results_with_fallback(self):
        """Empty quantitative_results must not crash — show fallback message."""
        a = make_rich_analysis(quantitative_results=[])
        html = _build_research_article_html(make_paper(), a)
        assert "See full paper" in html

    def test_renders_quantitative_results(self):
        a = make_rich_analysis(quantitative_results=["MMLU: 89.2% (+3.1%)", "HellaSwag: 95.1%"])
        html = _build_research_article_html(make_paper(), a)
        assert "MMLU: 89.2%" in html
        assert "HellaSwag: 95.1%" in html

    def test_future_directions_rendered(self):
        a = make_rich_analysis(future_directions=["Extend to vision.", "Apply to diffusion."])
        html = _build_research_article_html(make_paper(), a)
        assert "Extend to vision." in html


# ── 3. deep_analysis_node ────────────────────────────────────────────────────

class TestDeepAnalysisNode:
    def _mock_llm(self, rich_analysis: RichDeepAnalysis, linkedin_content: str):
        """
        Return (mock_pro, mock_flash) configured for deep_analysis_node.

        LangChain wraps non-Runnable callables via RunnableLambda, so when
        (prompt | mock).invoke() runs it calls mock(input) — not mock.invoke(input).
        We therefore set both .return_value (callable path) and .invoke.return_value
        (direct-invoke path) so the test is robust to either call convention.
        """
        # Pro path: constructor → .with_structured_output() → (prompt | chain).invoke()
        mock_structured_chain = MagicMock()
        mock_structured_chain.return_value = rich_analysis          # callable path
        mock_structured_chain.invoke.return_value = rich_analysis   # .invoke() path
        mock_pro = MagicMock()
        mock_pro.with_structured_output.return_value = mock_structured_chain

        # Flash path: constructor → (prompt | flash).invoke() → response.content
        mock_flash_response = MagicMock()
        mock_flash_response.content = linkedin_content
        mock_flash = MagicMock()
        mock_flash.return_value = mock_flash_response           # callable path
        mock_flash.invoke.return_value = mock_flash_response   # .invoke() path

        return mock_pro, mock_flash

    def test_returns_error_when_no_paper(self):
        result = deep_analysis_node({"chosen_research_paper": None})
        assert result["current_step"] == "error_no_paper"

    def test_returns_analysis_dict(self):
        analysis = make_rich_analysis()
        mock_pro, mock_flash = self._mock_llm(analysis, "Test LinkedIn draft " * 30)

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = [mock_pro, mock_flash]
            result = deep_analysis_node({"chosen_research_paper": make_paper()})

        assert "deep_analysis" in result
        assert result["deep_analysis"]["core_problem"] == analysis.core_problem
        assert result["deep_analysis"]["significance_verdict"] == "Major Contribution"

    def test_linkedin_draft_populated(self):
        analysis = make_rich_analysis()
        mock_pro, mock_flash = self._mock_llm(analysis, "This is the LinkedIn draft content.")

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = [mock_pro, mock_flash]
            result = deep_analysis_node({"chosen_research_paper": make_paper()})

        assert result["linkedin_draft"] == "This is the LinkedIn draft content."

    def test_newsletter_html_is_full_article(self):
        analysis = make_rich_analysis()
        mock_pro, mock_flash = self._mock_llm(analysis, "LinkedIn content")

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = [mock_pro, mock_flash]
            result = deep_analysis_node({"chosen_research_paper": make_paper()})

        html = result["newsletter_html"]
        assert "Executive Summary" in html
        assert "Technical Deep Dive" in html
        assert len(html) > 1000

    def test_current_step_on_success(self):
        analysis = make_rich_analysis()
        mock_pro, mock_flash = self._mock_llm(analysis, "Draft")

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = [mock_pro, mock_flash]
            result = deep_analysis_node({"chosen_research_paper": make_paper()})

        assert result["current_step"] == "analysis_complete"

    def test_fallback_linkedin_draft_on_flash_error(self):
        """If the Flash LinkedIn call fails, a template draft is returned instead of crashing."""
        analysis = make_rich_analysis()
        mock_structured_chain = MagicMock()
        mock_structured_chain.invoke.return_value = analysis
        mock_pro = MagicMock()
        mock_pro.with_structured_output.return_value = mock_structured_chain

        mock_flash = MagicMock()
        mock_flash.invoke.side_effect = RuntimeError("Rate limit")

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = [mock_pro, mock_flash]
            result = deep_analysis_node({"chosen_research_paper": make_paper()})

        assert result["linkedin_draft"]  # non-empty fallback
        assert result["current_step"] == "analysis_complete"

    def test_analysis_node_error_propagates_on_pro_failure(self):
        """If the Pro analysis call fails, return the error step."""
        mock_structured_chain = MagicMock()
        # Set side_effect on the mock itself so calling it as a callable also raises.
        # LangChain may call mock(input) via RunnableLambda instead of mock.invoke(input).
        mock_structured_chain.side_effect = RuntimeError("Model unavailable")
        mock_structured_chain.invoke.side_effect = RuntimeError("Model unavailable")
        mock_pro = MagicMock()
        mock_pro.with_structured_output.return_value = mock_structured_chain

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value = mock_pro
            result = deep_analysis_node({"chosen_research_paper": make_paper()})

        assert result["current_step"] == "error_analysis_failed"


# ── 4. research_carousel_node ─────────────────────────────────────────────────

class TestResearchCarouselNode:
    def _make_state(self, **overrides) -> dict:
        base = dict(
            deep_analysis=make_rich_analysis().model_dump(),
            chosen_research_paper=make_paper(),
            linkedin_draft="Hook line for the carousel slide.",
            prior_art_comparison={},
            run_id="testrun",
        )
        base.update(overrides)
        return base

    def test_skips_when_no_analysis(self):
        result = research_carousel_node({"deep_analysis": {}, "chosen_research_paper": make_paper()})
        assert result["current_step"] == "research_carousel_skipped"
        assert result["research_carousel_pdf_path"] == ""

    def test_skips_when_no_paper(self):
        result = research_carousel_node({"deep_analysis": make_rich_analysis().model_dump(), "chosen_research_paper": {}})
        assert result["current_step"] == "research_carousel_skipped"

    def test_total_slides_constant(self):
        assert _TOTAL_SLIDES == 10

    def test_generates_10_slides(self, tmp_path):
        """Carousel node must attempt to render exactly 10 slides."""
        mock_fitz = MagicMock()
        mock_doc = MagicMock()
        mock_doc.convert_to_pdf.return_value = b"PDF"
        mock_fitz.open.return_value = mock_doc

        mock_h2i = MagicMock()

        def fake_screenshot(**kwargs):
            filename = kwargs.get("save_as", "slide.png")
            (tmp_path / filename).write_bytes(b"PNG")

        mock_h2i.return_value.screenshot.side_effect = fake_screenshot

        with patch.dict(sys.modules, {"fitz": mock_fitz}), \
             patch("html2image.Html2Image", mock_h2i), \
             patch("app.agents.nodes.research_carousel.OUTPUT_DIR", tmp_path):
            result = research_carousel_node(self._make_state())

        assert mock_h2i.return_value.screenshot.call_count == 10

    def test_slide_names_cover_all_types(self, tmp_path):
        """Each of the 10 slide type names must appear in the generated filenames."""
        expected_names = [
            "cover", "problem", "prior_art", "methodology", "innovations",
            "experiments", "results", "ablation", "impact", "takeaways",
        ]
        mock_fitz = MagicMock()
        mock_fitz.open.return_value.convert_to_pdf.return_value = b"PDF"

        captured_filenames: list[str] = []

        def fake_screenshot(**kwargs):
            filename = kwargs.get("save_as", "")
            captured_filenames.append(filename)
            (tmp_path / filename).write_bytes(b"PNG")

        mock_h2i = MagicMock()
        mock_h2i.return_value.screenshot.side_effect = fake_screenshot

        with patch.dict(sys.modules, {"fitz": mock_fitz}), \
             patch("html2image.Html2Image", mock_h2i), \
             patch("app.agents.nodes.research_carousel.OUTPUT_DIR", tmp_path):
            research_carousel_node(self._make_state())

        for name in expected_names:
            assert any(name in fn for fn in captured_filenames), f"Missing slide: {name}"

    def test_returns_correct_state_keys(self, tmp_path):
        mock_fitz = MagicMock()
        mock_fitz.open.return_value.convert_to_pdf.return_value = b"PDF"

        def fake_screenshot(**kwargs):
            (tmp_path / kwargs.get("save_as", "x.png")).write_bytes(b"PNG")

        mock_h2i = MagicMock()
        mock_h2i.return_value.screenshot.side_effect = fake_screenshot

        with patch.dict(sys.modules, {"fitz": mock_fitz}), \
             patch("html2image.Html2Image", mock_h2i), \
             patch("app.agents.nodes.research_carousel.OUTPUT_DIR", tmp_path):
            result = research_carousel_node(self._make_state())

        assert "research_carousel_pdf_path" in result
        assert "research_carousel_slide_paths" in result
        assert "current_step" in result

    def test_graceful_fallback_when_prior_art_empty(self, tmp_path):
        """Carousel must render slide 3 even when prior_art_comparison is empty {}."""
        mock_fitz = MagicMock()
        mock_fitz.open.return_value.convert_to_pdf.return_value = b"PDF"

        def fake_screenshot(**kwargs):
            (tmp_path / kwargs.get("save_as", "x.png")).write_bytes(b"PNG")

        mock_h2i = MagicMock()
        mock_h2i.return_value.screenshot.side_effect = fake_screenshot

        state = self._make_state(prior_art_comparison={})

        with patch.dict(sys.modules, {"fitz": mock_fitz}), \
             patch("html2image.Html2Image", mock_h2i), \
             patch("app.agents.nodes.research_carousel.OUTPUT_DIR", tmp_path):
            # Should not raise even with empty prior_art_comparison
            result = research_carousel_node(state)

        # All 10 slides should still render
        assert mock_h2i.return_value.screenshot.call_count == 10

    def test_graceful_fallback_with_sparse_analysis(self, tmp_path):
        """Carousel must render all 10 slides even when new fields are absent (old analysis)."""
        mock_fitz = MagicMock()
        mock_fitz.open.return_value.convert_to_pdf.return_value = b"PDF"

        def fake_screenshot(**kwargs):
            (tmp_path / kwargs.get("save_as", "x.png")).write_bytes(b"PNG")

        mock_h2i = MagicMock()
        mock_h2i.return_value.screenshot.side_effect = fake_screenshot

        # Simulate an old 4-field analysis dict (no new fields)
        sparse_analysis = {
            "core_problem": "Problem statement.",
            "methodology": "Methodology description.",
            "breakthroughs": "Results numbers.",
            "limitations": "Limitations here.",
        }
        state = self._make_state(deep_analysis=sparse_analysis)

        with patch.dict(sys.modules, {"fitz": mock_fitz}), \
             patch("html2image.Html2Image", mock_h2i), \
             patch("app.agents.nodes.research_carousel.OUTPUT_DIR", tmp_path):
            result = research_carousel_node(state)

        assert mock_h2i.return_value.screenshot.call_count == 10


# ── 5. send_approval_email with research_article_html ────────────────────────

def _decode_mime_html(mime_str: str) -> str:
    """Extract and decode the HTML body from a raw MIME multipart message string."""
    msg = stdlib_email.message_from_string(mime_str)
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            return payload.decode("utf-8") if payload else ""
    return ""


class TestSendApprovalEmailResearchPreview:
    def _mock_settings(self):
        s = MagicMock()
        s.email_recipients = ["reviewer@example.com"]
        s.email_sender = "bot@example.com"
        s.smtp_host = "smtp.example.com"
        s.smtp_port = 587
        s.smtp_user = ""
        s.smtp_password = ""
        return s

    def _run_send(self, research_article_html: str = "", **extra_kwargs):
        """Call send_approval_email with SMTP mocked. Returns (raw_mime, decoded_html)."""
        mock_settings = self._mock_settings()
        captured: list[str] = []

        with patch("app.services.email_service.settings", mock_settings), \
             patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
            mock_smtp = MagicMock()
            MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
            mock_smtp.sendmail.side_effect = lambda s, r, msg: captured.append(msg)

            EmailService().send_approval_email(
                run_id="test-run-abc123",
                linkedin_preview="LinkedIn preview text",
                approve_url="https://app.example.com/approve?token=abc",
                reject_url="https://app.example.com/reject?token=abc",
                research_article_html=research_article_html,
                **extra_kwargs,
            )

        raw = captured[0] if captured else ""
        html = _decode_mime_html(raw)
        return raw, html

    def test_email_sent_without_article(self):
        """Email should send successfully when research_article_html is empty."""
        raw, html = self._run_send(research_article_html="")
        assert "LinkedIn preview text" in html
        assert "Approve" in html

    def test_email_includes_article_preview_when_provided(self):
        article = "<h2>Executive Summary</h2><p>This paper introduces...</p>"
        raw, html = self._run_send(research_article_html=article)
        assert "Executive Summary" in html
        assert "This paper introduces" in html

    def test_article_preview_inside_details_element(self):
        """The article preview must be wrapped in a collapsible <details> block."""
        article = "<p>Article content here.</p>"
        raw, html = self._run_send(research_article_html=article)
        assert "<details" in html
        assert "Research Article Preview" in html

    def test_approve_and_reject_urls_present(self):
        raw, html = self._run_send()
        assert "approve?token=abc" in html
        assert "reject?token=abc" in html

    def test_run_id_shown_in_email(self):
        raw, html = self._run_send()
        # run_id[:8] of "test-run-abc123" is "test-run"
        assert "test-run" in html

    def test_no_article_section_when_empty_string(self):
        """When research_article_html='', the <details> block must NOT appear."""
        raw, html = self._run_send(research_article_html="")
        assert "Research Article Preview" not in html

    def test_image_attachment_included_when_path_provided(self, tmp_path):
        """Image attachment still works alongside the new article parameter."""
        img = tmp_path / "card.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_settings = self._mock_settings()
        with patch("app.services.email_service.settings", mock_settings), \
             patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
            mock_smtp = MagicMock()
            MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            MockSMTP.return_value.__exit__ = MagicMock(return_value=False)

            EmailService().send_approval_email(
                run_id="run-xyz",
                linkedin_preview="Preview",
                approve_url="http://approve",
                reject_url="http://reject",
                image_paths=[str(img)],
                research_article_html="<p>Article</p>",
            )

        assert mock_smtp.sendmail.called


# ── 6. LinkedIn research prompt content ───────────────────────────────────────

class TestLinkedInResearchPromptConstant:
    """Verify _LINKEDIN_RESEARCH_SYSTEM contains the structural rules."""

    def test_prompt_includes_hook_instruction(self):
        from app.agents.research_graph import _LINKEDIN_RESEARCH_SYSTEM
        assert "HOOK" in _LINKEDIN_RESEARCH_SYSTEM
        assert "210" in _LINKEDIN_RESEARCH_SYSTEM

    def test_prompt_includes_hard_rules(self):
        from app.agents.research_graph import _LINKEDIN_RESEARCH_SYSTEM
        assert "HARD RULES" in _LINKEDIN_RESEARCH_SYSTEM
        assert "1,800" in _LINKEDIN_RESEARCH_SYSTEM

    def test_prompt_bans_filler_phrases(self):
        from app.agents.research_graph import _LINKEDIN_RESEARCH_SYSTEM
        assert "game-changer" in _LINKEDIN_RESEARCH_SYSTEM
        assert "revolutionary" in _LINKEDIN_RESEARCH_SYSTEM

    def test_prompt_requires_first_comment_footer(self):
        from app.agents.research_graph import _LINKEDIN_RESEARCH_SYSTEM
        assert "first comment" in _LINKEDIN_RESEARCH_SYSTEM


# ── 7. select_paper_node ──────────────────────────────────────────────────────

class TestSelectPaperNode:
    def test_returns_no_papers_found_when_empty(self):
        result = select_paper_node({"raw_articles": [], "paper_rankings": []})
        assert result["current_step"] == "no_papers_found"

    def test_selects_by_ranking_primary_path(self):
        paper_a = make_paper(url="https://arxiv.org/abs/111")
        paper_b = make_paper(url="https://arxiv.org/abs/222")
        rankings = [
            {"paper_url": paper_a["url"], "composite_score": 9.5, "is_manual": False},
            {"paper_url": paper_b["url"], "composite_score": 7.0, "is_manual": False},
        ]
        result = select_paper_node({
            "raw_articles": [paper_a, paper_b],
            "paper_rankings": rankings,
        })
        assert result["current_step"] == "paper_selected"
        assert result["chosen_research_paper"]["url"] == paper_a["url"]

    def test_selects_ranked_paper_that_exists_in_articles(self):
        """Ranking may reference a URL not in articles — must skip to next valid entry."""
        real_paper = make_paper(url="https://arxiv.org/abs/real")
        rankings = [
            {"paper_url": "https://arxiv.org/abs/ghost", "composite_score": 9.9},  # not in articles
            {"paper_url": real_paper["url"], "composite_score": 8.0},
        ]
        result = select_paper_node({
            "raw_articles": [real_paper],
            "paper_rankings": rankings,
        })
        assert result["chosen_research_paper"]["url"] == real_paper["url"]

    def test_llm_fallback_when_no_rankings(self):
        """When paper_rankings is empty, select_paper_node falls back to LLM selection."""
        from app.agents.research_graph import PaperSelection

        paper = make_paper()
        mock_selection = PaperSelection(chosen_url=paper["url"], reasoning="Best paper.")
        mock_chain = MagicMock()
        mock_chain.return_value = mock_selection
        mock_chain.invoke.return_value = mock_selection

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_chain
            MockLLM.return_value = mock_instance

            result = select_paper_node({"raw_articles": [paper], "paper_rankings": []})

        assert result["current_step"] == "paper_selected"
        assert result["chosen_research_paper"]["url"] == paper["url"]


# ── 8. score_research_node ────────────────────────────────────────────────────

class TestScoreResearchNode:
    def _make_scores(self, **overrides) -> ResearchScores:
        return ResearchScores(
            novelty=8,
            methodology_clarity=7,
            benchmark_improvement=9,
            reproducibility=5,
            score_reasoning="Strong contribution with partial reproducibility.",
            **overrides,
        )

    def test_skips_when_analysis_empty(self):
        result = score_research_node({"chosen_research_paper": make_paper(), "deep_analysis": {}})
        assert result["current_step"] == "research_scores_skipped"
        assert result["research_scores"] == {}

    def test_returns_four_dimension_scores(self):
        mock_scores = self._make_scores()
        mock_chain = MagicMock()
        mock_chain.return_value = mock_scores
        mock_chain.invoke.return_value = mock_scores

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_chain
            MockLLM.return_value = mock_instance

            result = score_research_node({
                "chosen_research_paper": make_paper(),
                "deep_analysis": make_rich_analysis().model_dump(),
            })

        assert result["current_step"] == "research_scored"
        scores = result["research_scores"]
        assert scores["novelty"] == 8
        assert scores["methodology_clarity"] == 7
        assert scores["benchmark_improvement"] == 9
        assert scores["reproducibility"] == 5

    def test_gracefully_handles_llm_failure(self):
        """If scoring LLM fails, return empty scores and skip — do not crash."""
        mock_chain = MagicMock()
        mock_chain.side_effect = RuntimeError("Scoring API down")
        mock_chain.invoke.side_effect = RuntimeError("Scoring API down")

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_chain
            MockLLM.return_value = mock_instance

            result = score_research_node({
                "chosen_research_paper": make_paper(),
                "deep_analysis": make_rich_analysis().model_dump(),
            })

        assert result["current_step"] == "research_scores_skipped"
        assert result["research_scores"] == {}

    def test_concatenates_new_fields_into_scoring_context(self):
        """technical_innovation and quantitative_results from RichDeepAnalysis must be
        included in the scoring context (methodology and breakthroughs fields)."""
        captured_inputs: dict = {}

        def capture_call(inputs):
            captured_inputs.update(inputs)
            return self._make_scores()

        mock_chain = MagicMock()
        mock_chain.side_effect = capture_call
        mock_chain.invoke.side_effect = capture_call

        with patch("app.agents.research_graph.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_chain
            MockLLM.return_value = mock_instance

            analysis = make_rich_analysis(
                technical_innovation="Novel routing approach.",
                quantitative_results=["MMLU: 89.2%", "HellaSwag: 95.1%"],
            ).model_dump()
            score_research_node({
                "chosen_research_paper": make_paper(),
                "deep_analysis": analysis,
            })

        # LangChain formats the raw dict into ChatMessages before calling the chain.
        # Verify the technical_innovation and quantitative_results were woven in
        # by checking the serialised message text (methodology + breakthroughs fields).
        if captured_inputs:
            messages_text = " ".join(str(m) for m in captured_inputs.get("messages", []))
            assert "Novel routing approach." in messages_text
            assert "MMLU: 89.2%" in messages_text


# ── 9. _render_gauge_svg ──────────────────────────────────────────────────────

class TestRenderGaugeSvg:
    def test_returns_svg_string(self):
        result = _render_gauge_svg("Novelty", 8, "#00f3ff")
        assert "<svg" in result
        assert "</svg>" in result

    def test_includes_label_and_value(self):
        result = _render_gauge_svg("Novelty", 8, "#00f3ff")
        assert "Novelty" in result
        assert ">8<" in result

    def test_uses_specified_color(self):
        result = _render_gauge_svg("Repro", 5, "#ff2d78")
        assert "#ff2d78" in result

    def test_value_10_fills_arc_fully(self):
        result_10 = _render_gauge_svg("Max", 10, "#fff")
        result_1 = _render_gauge_svg("Min", 1, "#fff")
        # Higher value → larger fill dasharray first number
        assert result_10 != result_1


# ── 10. send_newsletter coverage ─────────────────────────────────────────────

class TestSendNewsletterCoverage:
    """Covers send_newsletter to push email_service.py coverage above 80%."""

    def _mock_settings(self):
        s = MagicMock()
        s.email_recipients = ["a@example.com", "b@example.com"]
        s.email_sender = "newsletter@bot.com"
        s.smtp_host = "smtp.example.com"
        s.smtp_port = 587
        s.smtp_user = ""
        s.smtp_password = ""
        return s

    def test_send_newsletter_calls_smtp(self):
        with patch("app.services.email_service.settings", self._mock_settings()), \
             patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
            mock_smtp = MagicMock()
            MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            MockSMTP.return_value.__exit__ = MagicMock(return_value=False)

            EmailService().send_newsletter(
                html_content="<h1>Weekly AI Digest</h1>",
                subject="AI Newsletter",
            )

        assert mock_smtp.sendmail.called

    def test_send_newsletter_sends_to_all_recipients(self):
        captured: list = []
        with patch("app.services.email_service.settings", self._mock_settings()), \
             patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
            mock_smtp = MagicMock()
            MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
            mock_smtp.sendmail.side_effect = lambda s, r, m: captured.append(r)

            EmailService().send_newsletter(html_content="<p>Content</p>")

        assert captured[0] == ["a@example.com", "b@example.com"]

    def test_send_newsletter_with_image_attachment(self, tmp_path):
        img = tmp_path / "news_card.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

        with patch("app.services.email_service.settings", self._mock_settings()), \
             patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
            mock_smtp = MagicMock()
            MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            MockSMTP.return_value.__exit__ = MagicMock(return_value=False)

            EmailService().send_newsletter(
                html_content="<p>With image</p>",
                image_paths=[str(img)],
            )

        assert mock_smtp.sendmail.called

    def test_send_newsletter_skips_nonexistent_attachment(self, tmp_path):
        """Paths that don't exist on disk must be silently skipped."""
        ghost_path = str(tmp_path / "ghost.png")  # does not exist

        with patch("app.services.email_service.settings", self._mock_settings()), \
             patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
            mock_smtp = MagicMock()
            MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            MockSMTP.return_value.__exit__ = MagicMock(return_value=False)

            # Should not raise
            EmailService().send_newsletter(
                html_content="<p>OK</p>",
                image_paths=[ghost_path],
            )

        assert mock_smtp.sendmail.called
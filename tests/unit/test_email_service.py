"""
Tests for the fast-review approval email (thumbnail grid + prominent CTA).

Background: `send_approval_email` is shared by both the news pipeline and the
research pipeline (called from the single `human_approval_node` in
app/agents/nodes/approval.py). The news pipeline never populates
`research_carousel_slide_paths`, so it never passes `slide_image_paths` to
this function — these tests prove that path renders exactly the old
full-dump layout, while the research pipeline's new `slide_image_paths`
argument activates the new fast-review thumbnail grid additively, without
disturbing the existing LinkedIn preview / article / approve-reject sections.
"""

from __future__ import annotations

import email as stdlib_email
from unittest.mock import MagicMock, patch

from app.services.email_service import EmailService


def _decode_mime_html(mime_str: str) -> str:
    """Extract and decode the HTML body from a raw MIME multipart message string."""
    msg = stdlib_email.message_from_string(mime_str)
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            return payload.decode("utf-8") if payload else ""
    return ""


def _mock_settings():
    s = MagicMock()
    s.email_recipients = ["reviewer@example.com"]
    s.email_sender = "bot@example.com"
    s.smtp_host = "smtp.example.com"
    s.smtp_port = 587
    s.smtp_user = ""
    s.smtp_password = ""
    s.app_base_url = "https://app.example.com"
    return s


def _run_send(**kwargs) -> tuple[str, str]:
    """Call send_approval_email with SMTP mocked. Returns (raw_mime, decoded_html)."""
    mock_settings = _mock_settings()
    captured: list[str] = []

    with patch("app.services.email_service.settings", mock_settings), \
         patch("app.services.email_service.smtplib.SMTP") as MockSMTP:
        mock_smtp = MagicMock()
        MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_smtp.sendmail.side_effect = lambda s, r, msg: captured.append(msg)

        defaults = dict(
            run_id="run-abc12345",
            linkedin_preview="LinkedIn preview text",
            approve_url="https://app.example.com/approve?token=abc",
            reject_url="https://app.example.com/reject?token=abc",
        )
        defaults.update(kwargs)
        EmailService().send_approval_email(**defaults)

    raw = captured[0] if captured else ""
    html = _decode_mime_html(raw)
    return raw, html


# ── News pipeline: unchanged format ───────────────────────────────────────────

class TestNewsPipelineEmailUnchanged:
    """
    The news pipeline's call site never passes `slide_image_paths` (it doesn't
    populate `research_carousel_slide_paths` in state), so these calls must
    keep producing the original full-dump layout untouched.
    """

    def test_news_shaped_call_has_no_grid_section(self):
        """Calling without slide_image_paths (the news pipeline's call shape)."""
        _, html = _run_send(image_paths=None, research_article_html="")
        assert "Slide Preview" not in html
        assert "carousel slides generated" not in html

    def test_news_shaped_call_keeps_old_bottom_buttons(self):
        _, html = _run_send()
        assert "Approve &amp; Publish" in html
        assert "Reject &amp; Revise" in html
        assert "approve?token=abc" in html
        assert "reject?token=abc" in html

    def test_explicit_empty_slide_list_also_keeps_old_layout(self):
        """An explicit empty list must behave identically to omitting the arg."""
        _, html = _run_send(slide_image_paths=[])
        assert "Slide Preview" not in html

    def test_linkedin_preview_still_present(self):
        _, html = _run_send()
        assert "LinkedIn preview text" in html
        assert "LinkedIn Post Preview" in html

    def test_run_id_still_shown(self):
        _, html = _run_send(run_id="news-run-99887766")
        assert "news-run" in html


# ── Research pipeline: new thumbnail grid ─────────────────────────────────────

class TestResearchPipelineThumbnailGrid:
    """
    The research pipeline's call site (app/agents/nodes/approval.py) passes
    `slide_image_paths=list(research_slides)` sourced from the
    research-specific `research_carousel_slide_paths` state key.
    """

    SLIDES = ["/tmp/slides/slide_01.png", "/tmp/slides/slide_02.png", "/tmp/slides/slide_03.png"]

    def test_grid_section_rendered_when_slides_present(self):
        _, html = _run_send(slide_image_paths=self.SLIDES)
        assert "Slide Preview" in html
        assert "3 carousel slides generated" in html

    def test_each_slide_rendered_as_public_url_thumbnail(self):
        _, html = _run_send(slide_image_paths=self.SLIDES)
        for name in ("slide_01.png", "slide_02.png", "slide_03.png"):
            assert f"https://app.example.com/static/images/{name}" in html

    def test_prominent_cta_appears_before_linkedin_preview(self):
        """The top-of-email prominent Approve & Publish button must precede
        the LinkedIn preview section (fast-scan-first layout)."""
        _, html = _run_send(slide_image_paths=self.SLIDES)
        grid_cta_index = html.index("carousel slides generated")
        linkedin_index = html.index("LinkedIn Post Preview")
        assert grid_cta_index < linkedin_index

    def test_grid_still_includes_approve_and_reject_links(self):
        _, html = _run_send(slide_image_paths=self.SLIDES)
        assert html.count("approve?token=abc") >= 1
        assert html.count("reject?token=abc") >= 1

    def test_existing_sections_still_present_alongside_grid(self):
        """New grid is additive: article/LinkedIn/bottom-button sections stay."""
        _, html = _run_send(
            slide_image_paths=self.SLIDES,
            research_article_html="<h2>Executive Summary</h2>",
        )
        assert "Slide Preview" in html
        assert "Executive Summary" in html
        assert "LinkedIn Post Preview" in html

    def test_no_grid_rendered_for_empty_slide_urls_after_conversion(self):
        """Defensive: an empty input list produces no grid section at all."""
        _, html = _run_send(slide_image_paths=[])
        assert "Slide Preview" not in html

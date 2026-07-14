"""
Tests for the carousel story-image fetch pipeline and the shared
capture-with-retry helper used by all html2image call sites.
"""

from __future__ import annotations

from app.agents.nodes.image_gen import _extract_body_image_url, _extract_meta_image_url
from app.agents.nodes.screenshot_utils import capture_slide


# ── _extract_meta_image_url ─────────────────────────────────────────────────

class TestExtractMetaImageUrl:
    def test_finds_og_image_property_then_content(self):
        html = '<meta property="og:image" content="https://example.com/photo.jpg">'
        assert _extract_meta_image_url(html) == "https://example.com/photo.jpg"

    def test_finds_og_image_content_then_property(self):
        html = '<meta content="https://example.com/photo.jpg" property="og:image">'
        assert _extract_meta_image_url(html) == "https://example.com/photo.jpg"

    def test_falls_back_to_twitter_image(self):
        html = '<meta name="twitter:image" content="https://example.com/tw.jpg">'
        assert _extract_meta_image_url(html) == "https://example.com/tw.jpg"

    def test_falls_back_to_link_image_src(self):
        html = '<link rel="image_src" href="https://example.com/linked.jpg">'
        assert _extract_meta_image_url(html) == "https://example.com/linked.jpg"

    def test_skips_logo_candidate_and_uses_next_tier(self):
        html = (
            '<meta property="og:image" content="https://example.com/site-logo.png">'
            '<meta name="twitter:image" content="https://example.com/real-photo.jpg">'
        )
        assert _extract_meta_image_url(html) == "https://example.com/real-photo.jpg"

    def test_returns_none_when_no_tags_present(self):
        assert _extract_meta_image_url("<html><body>no meta here</body></html>") is None


# ── _extract_body_image_url ─────────────────────────────────────────────────

class TestExtractBodyImageUrl:
    def test_finds_substantial_body_image(self):
        html = '<article><img src="https://example.com/story.jpg" width="800" height="450"></article>'
        assert _extract_body_image_url(html) == "https://example.com/story.jpg"

    def test_skips_small_icon_dimensions(self):
        html = (
            '<img src="https://example.com/icon.png" width="32" height="32">'
            '<img src="https://example.com/photo.jpg" width="600" height="400">'
        )
        assert _extract_body_image_url(html) == "https://example.com/photo.jpg"

    def test_skips_logo_named_src_regardless_of_size(self):
        html = '<img src="https://example.com/company-logo.png" width="900" height="900">'
        assert _extract_body_image_url(html) is None

    def test_skips_data_uri_images(self):
        html = '<img src="data:image/png;base64,abc123">'
        assert _extract_body_image_url(html) is None

    def test_falls_back_to_data_src_lazy_load_attribute(self):
        html = '<img data-src="https://example.com/lazy.jpg" width="640" height="360">'
        assert _extract_body_image_url(html) == "https://example.com/lazy.jpg"


# ── capture_slide ────────────────────────────────────────────────────────────

class _FakeHti:
    """Stub matching the subset of Html2Image used by capture_slide."""

    def __init__(self, output_path, successes: set[int]):
        self.output_path = output_path
        self._successes = successes
        self.calls = 0

    def screenshot(self, html_str, save_as):
        self.calls += 1
        if self.calls in self._successes:
            (self.output_path / save_as).write_bytes(b"fake-png")


class TestCaptureSlide:
    def test_returns_path_when_first_attempt_succeeds(self, tmp_path):
        hti = _FakeHti(tmp_path, successes={1})
        result = capture_slide(hti, "<html></html>", "slide.png", label="cover", output_dir=tmp_path)
        assert result == str(tmp_path / "slide.png")
        assert hti.calls == 1

    def test_retries_once_and_recovers_from_transient_crash(self, tmp_path):
        hti = _FakeHti(tmp_path, successes={2})
        result = capture_slide(hti, "<html></html>", "slide.png", label="story_0", output_dir=tmp_path)
        assert result == str(tmp_path / "slide.png")
        assert hti.calls == 2

    def test_gives_up_after_two_failures_and_returns_none(self, tmp_path):
        hti = _FakeHti(tmp_path, successes=set())
        result = capture_slide(hti, "<html></html>", "slide.png", label="closing", output_dir=tmp_path)
        assert result is None
        assert hti.calls == 2

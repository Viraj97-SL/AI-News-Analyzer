"""
Tests for figure_quality.py — luminance/dominance screening of extracted
paper figures, with auto-contrast/inversion recovery before rejection.
"""

from __future__ import annotations

import io

from PIL import Image, ImageStat

from app.agents.nodes.figure_quality import assess_and_correct

MIN_LUMINANCE = 40.0
MAX_DOMINANT_RATIO = 0.75


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _checkerboard(size: int, value_a: int, value_b: int) -> Image.Image:
    img = Image.new("L", (size, size), value_a)
    pixels = img.load()
    for x in range(size):
        for y in range(size):
            if (x + y) % 2 == 0:
                pixels[x, y] = value_b
    return img.convert("RGB")


class TestAssessAndCorrect:
    def test_readable_image_passes_without_correction(self):
        img = _checkerboard(40, 100, 180)
        result = assess_and_correct(_png_bytes(img), MIN_LUMINANCE, MAX_DOMINANT_RATIO)
        assert result.passed
        assert result.corrected_bytes is None

    def test_dark_but_varied_image_recovered_by_autocontrast(self):
        """Near-black figure with real signal (not uniform) must pass after autocontrast."""
        img = _checkerboard(40, 5, 60)
        raw_luminance = ImageStat.Stat(img.convert("L")).mean[0]
        assert raw_luminance < MIN_LUMINANCE  # confirms it would have failed raw

        result = assess_and_correct(_png_bytes(img), MIN_LUMINANCE, MAX_DOMINANT_RATIO)
        assert result.passed
        assert result.corrected_bytes is not None
        # the reported metrics describe the corrected image actually returned, not the raw one
        assert result.mean_luminance >= MIN_LUMINANCE

    def test_solid_near_black_image_rejected(self):
        """A truly uniform dark image (empty box) has no signal to recover — must be rejected."""
        img = Image.new("RGB", (40, 40), (10, 10, 10))
        result = assess_and_correct(_png_bytes(img), MIN_LUMINANCE, MAX_DOMINANT_RATIO)
        assert not result.passed
        assert result.corrected_bytes is None

    def test_solid_white_image_rejected_by_dominance(self):
        """High luminance alone isn't enough — a single dominant color still fails."""
        img = Image.new("RGB", (40, 40), (250, 250, 250))
        result = assess_and_correct(_png_bytes(img), MIN_LUMINANCE, MAX_DOMINANT_RATIO)
        assert not result.passed

    def test_mean_luminance_and_dominance_reported_for_passing_image(self):
        img = _checkerboard(40, 100, 180)
        result = assess_and_correct(_png_bytes(img), MIN_LUMINANCE, MAX_DOMINANT_RATIO)
        assert 100 <= result.mean_luminance <= 180
        assert 0.0 <= result.dominant_color_ratio <= 1.0

    def test_configurable_thresholds_change_outcome(self):
        """The same image can pass or fail depending on the configured threshold."""
        img = _checkerboard(40, 5, 60)
        strict = assess_and_correct(_png_bytes(img), min_luminance=250.0, max_dominant_ratio=0.1)
        lenient = assess_and_correct(_png_bytes(img), min_luminance=0.0, max_dominant_ratio=1.0)
        assert not strict.passed
        assert lenient.passed
        assert lenient.corrected_bytes is None  # already within the lenient threshold, no correction needed

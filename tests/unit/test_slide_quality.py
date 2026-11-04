"""Tests for the rendered-slide blank-space metric (research pipeline v2, phase 4)."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from app.agents.nodes.slide_quality import largest_blank_band, measure_fill_ratio

W, H = 540, 675  # half-size portrait canvas keeps tests fast


def _gradient_canvas() -> Image.Image:
    """Light vertical gradient like the slide background (must not count as content)."""
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        shade = 241 - int(10 * y / H)
        draw.line([(0, y), (W, y)], fill=(shade, shade + 4, 249))
    return img


def _text_block(img: Image.Image, top: int, bottom: int) -> None:
    draw = ImageDraw.Draw(img)
    for y in range(top, bottom, 14):
        draw.rectangle([40, y, W - 40, y + 7], fill=(15, 23, 42))


def test_full_slide_has_high_fill_ratio(tmp_path):
    img = _gradient_canvas()
    _text_block(img, 40, H - 40)
    path = tmp_path / "full.png"
    img.save(path)
    assert measure_fill_ratio(path) > 0.9


def test_top_heavy_slide_has_low_fill_ratio(tmp_path):
    img = _gradient_canvas()
    _text_block(img, 40, int(H * 0.35))  # content in top third, rest blank
    path = tmp_path / "top.png"
    img.save(path)
    assert measure_fill_ratio(path) < 0.6


def test_pale_panel_background_counts_as_blank(tmp_path):
    """A big empty pastel card (the old 'stretched panel' bug) is still blank space."""
    img = _gradient_canvas()
    _text_block(img, 40, 120)
    ImageDraw.Draw(img).rectangle(
        [30, 130, W - 30, H - 60], fill=(237, 233, 254), outline=(200, 200, 230)
    )
    path = tmp_path / "panel.png"
    img.save(path)
    assert measure_fill_ratio(path) < 0.6


def test_largest_blank_band_on_known_rows():
    rows = [True] * 10 + [False] * 30 + [True] * 5 + [False] * 5
    assert largest_blank_band(rows) == 30


def test_largest_blank_band_all_content():
    assert largest_blank_band([True] * 8) == 0


def test_measure_fill_ratio_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        measure_fill_ratio(tmp_path / "nope.png")

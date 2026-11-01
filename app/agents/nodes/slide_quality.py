"""
Blank-space metric for rendered carousel slides (research pipeline v2, phase 4).

`measure_fill_ratio` = 1 - (tallest run of empty rows inside the body area ÷ body
height). A slide whose content stops a third of the way down scores ~0.4; a
balanced slide scores > 0.85. Logged per run, stored in state as
`slide_fill_ratios`, and tracked by the drift report (`slide_fill_ratio_avg`).

A row is "empty" when almost no pixel stands out from that row's median colour,
so smooth background gradients and large pale panels both count as blank —
exactly the visual problem we want to catch — while text, borders of real
content, charts and figures count as ink.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Body area excludes the header strip and the footer/progress bar, which are always inked.
_BODY_TOP_FRAC = 0.06
_BODY_BOTTOM_FRAC = 0.88
_CHANNEL_DELTA = 48  # max per-channel distance from row median to count a pixel as ink
_MIN_INK_FRACTION = 0.01  # row needs ≥1% ink pixels to count as content
_ANALYSIS_WIDTH = 270  # downscale for speed; row structure survives


def largest_blank_band(row_has_ink: list[bool]) -> int:
    longest = current = 0
    for inked in row_has_ink:
        current = 0 if inked else current + 1
        longest = max(longest, current)
    return longest


def _row_has_ink(pixels: list[tuple[int, int, int]]) -> bool:
    channels = list(zip(*pixels, strict=True))
    median = tuple(sorted(ch)[len(ch) // 2] for ch in channels)
    ink = sum(1 for px in pixels if max(abs(px[i] - median[i]) for i in range(3)) > _CHANNEL_DELTA)
    return ink / len(pixels) >= _MIN_INK_FRACTION


def measure_fill_ratio(png_path: Path | str) -> float:
    path = Path(png_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as raw:
        img = raw.convert("RGB")
        scale = _ANALYSIS_WIDTH / img.width
        img = img.resize((_ANALYSIS_WIDTH, max(1, round(img.height * scale))))
    top = int(img.height * _BODY_TOP_FRAC)
    bottom = int(img.height * _BODY_BOTTOM_FRAC)
    width = img.width
    data = list(img.getdata())
    rows = [_row_has_ink(data[y * width : (y + 1) * width]) for y in range(top, bottom)]
    body_height = max(1, bottom - top)
    return round(1 - largest_blank_band(rows) / body_height, 3)

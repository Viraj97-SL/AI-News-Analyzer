"""
Figure quality filter for extracted paper figures.

PDF/HTML figure extraction has no notion of *readable* — a near-black LiDAR
point cloud or a mostly-white blank page scores fine on size/aspect-ratio
heuristics alone. This module screens candidates by grayscale mean
luminance and single-value pixel dominance, attempting auto-contrast then
inversion recovery before a figure is rejected outright.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageStat


@dataclass(frozen=True)
class FigureQualityResult:
    passed: bool
    mean_luminance: float  # 0-255 grayscale mean
    dominant_color_ratio: float  # fraction of pixels at the single most common grayscale value
    corrected_bytes: bytes | None  # auto-contrast/inverted PNG bytes, if correction was needed


def _luminance_and_dominance(gray_img: Image.Image) -> tuple[float, float]:
    stat = ImageStat.Stat(gray_img)
    mean_luminance = stat.mean[0]
    hist = gray_img.histogram()
    total = sum(hist)
    dominant_ratio = (max(hist) / total) if total else 1.0
    return mean_luminance, dominant_ratio


def assess_and_correct(
    image_bytes: bytes,
    min_luminance: float,
    max_dominant_ratio: float,
) -> FigureQualityResult:
    """
    Reject figures that are too dark or too uniform to read.

    Tries auto-contrast, then auto-contrast + inversion, before giving up —
    catches the common "near-black scientific plot" case where the data is
    present but rendered at the dark end of the range.
    """
    base_rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    luminance, dominance = _luminance_and_dominance(base_rgb.convert("L"))

    if luminance >= min_luminance and dominance <= max_dominant_ratio:
        return FigureQualityResult(True, luminance, dominance, None)

    candidates = (
        ImageOps.autocontrast(base_rgb, cutoff=1),
        ImageOps.invert(ImageOps.autocontrast(base_rgb, cutoff=1).convert("RGB")),
    )
    for candidate_rgb in candidates:
        cand_luminance, cand_dominance = _luminance_and_dominance(candidate_rgb.convert("L"))
        if cand_luminance >= min_luminance and cand_dominance <= max_dominant_ratio:
            buf = io.BytesIO()
            candidate_rgb.save(buf, format="PNG")
            return FigureQualityResult(True, cand_luminance, cand_dominance, buf.getvalue())

    return FigureQualityResult(False, luminance, dominance, None)

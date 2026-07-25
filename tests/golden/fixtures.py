"""
Synthetic paper fixtures for golden/snapshot rendering tests.

Each fixture pairs a `slide_type` with the exact Jinja context dict that
`research_carousel_node` (app/agents/nodes/research_carousel.py) would build
for that slide, so the golden test can render through the real template
without needing a full pipeline run or any network/LLM call.

The fixtures deliberately cover the three edge cases named in the testing
audit:
  - `figure_rich`: 2+ real paper figures (exercises the "figures" slide,
    which only appears when len(paper_figures) >= 2).
  - `abstract_only`: sparse/empty structured fields, mirroring a paper where
    only the abstract was available (see app/agents/nodes/full_text.py) —
    exercises the "See full paper" / methodology-fallback text paths.
  - `latex_title_cover` / `math_heavy_methodology`: LLM-produced strings
    containing raw LaTeX (`$...$`, `\text{...}`, bare `^`/`_`), run through
    `normalize_text`/`normalize_title` exactly as `select_paper_node` and
    `deep_analysis_node` do in production (see app/agents/research_graph.py),
    so the baseline captures the *cleaned* rendering and a future
    normalization regression shows up as a visual diff.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from app.agents.nodes.text_utils import normalize_text, normalize_title


def _synthetic_figure_b64(label: str, color: tuple[int, int, int]) -> str:
    """Build a small deterministic in-memory PNG (no network/disk asset needed)."""
    img = Image.new("RGB", (400, 300), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 380, 280], outline=color, width=6)
    draw.text((40, 130), label, fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass(frozen=True)
class GoldenFixture:
    """One synthetic paper scenario rendered as a single representative slide."""

    name: str
    slide_type: str
    context: dict[str, object] = field(default_factory=dict)


def _figure_rich() -> GoldenFixture:
    return GoldenFixture(
        name="figure_rich",
        slide_type="figures",
        context={
            "paper_figures": [
                {
                    "b64": _synthetic_figure_b64("Fig. 1: Architecture", (14, 165, 233)),
                    "caption": "Figure 1: Overall model architecture.",
                },
                {
                    "b64": _synthetic_figure_b64("Fig. 2: Ablation", (124, 58, 237)),
                    "caption": "Figure 2: Ablation results across components.",
                },
            ],
            "slide_num": 5,
            "total_slides": 11,
        },
    )


def _abstract_only() -> GoldenFixture:
    """Low-confidence paper: only the abstract was available (full_text_available=False).

    Mirrors _EMPTY_RESULT in app/agents/nodes/full_text.py — quantitative_results and
    claims_evidence are empty, so the "results" slide must fall back to the
    breakthroughs paragraph instead of crashing on an empty list.
    """
    return GoldenFixture(
        name="abstract_only",
        slide_type="results",
        context={
            "claims_evidence": [],
            "quantitative_results": [],
            "breakthroughs": (
                "The abstract reports improved accuracy over prior baselines, "
                "but does not include specific benchmark numbers in the excerpt available."
            ),
            "benchmark_chart_b64": "",
            "slide_num": 7,
            "total_slides": 9,
        },
    )


def _latex_title_cover() -> GoldenFixture:
    """LaTeX-in-title regression: raw arXiv title text leaking `$...\\text{...}$`.

    `select_paper_node` normalizes `chosen_research_paper["title"]` via
    `normalize_title` before it ever reaches the carousel (research_graph.py:234),
    so the fixture applies the same normalization here rather than passing the
    raw LaTeX through untouched.
    """
    raw_title = (
        "M$^\\text{4}$World: A Benchmark for Spatiotemporal Reasoning in Video-Language Models"
    )
    return GoldenFixture(
        name="latex_title_cover",
        slide_type="cover",
        context={
            "title": normalize_title(raw_title),
            "hook": normalize_text(
                "Current video-language models score below 40 percent accuracy on "
                "M$^\\text{4}$World's 4D reasoning tasks — humans hit 92 percent."
            ),
            "significance_verdict": "Major Contribution",
            "paper_url": "https://arxiv.org/abs/2401.11111",
            "is_classic_paper": False,
            "score_gauges_html": "",
            "slide_num": 1,
            "total_slides": 10,
        },
    )


def _math_heavy_methodology() -> GoldenFixture:
    """Math-heavy methodology text with bare/braced sub- and superscripts.

    Exercises normalize_text's digit-exponent and braced-subscript rewriting
    (O(n^2) -> O(n²), attention_i -> attentionᵢ-style patterns) on a
    body-copy slide rather than the title, since methodology text is the
    other common leak point for raw LaTeX.
    """
    raw_methodology = (
        "Standard attention costs O(n^2) memory. We replace it with a routed "
        "sparse variant Attn_{sparse} that reduces this to O(n log n), where "
        "each token i attends only to a learned subset S_i of size k << n."
    )
    return GoldenFixture(
        name="math_heavy_methodology",
        slide_type="methodology",
        context={
            "methodology": normalize_text(raw_methodology),
            "technical_innovation": normalize_text(
                "Unlike fixed local-window methods, S_i is learned per-token via a routing MLP."
            ),
            "architecture_diagram_b64": "",
            "architecture_spec_svg": "",
            "architecture_fallback_text": "",
            "slide_num": 4,
            "total_slides": 10,
        },
    )


GOLDEN_FIXTURES: list[GoldenFixture] = [
    _figure_rich(),
    _abstract_only(),
    _latex_title_cover(),
    _math_heavy_methodology(),
]

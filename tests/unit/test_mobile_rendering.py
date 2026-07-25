"""
Mobile-first rendering tests for the research carousel template.

LinkedIn carousels are viewed at roughly 350px wide on mobile feeds, but
nothing previously rendered a slide at that width and inspected the actual
computed font sizes — the reference failure case is a comparison/results
table that looked fine at desktop width but had illegibly small text on a
phone (see app/templates/research_carousel_slide.html, the "prior_art" and
"results" slide types).

html2image (this repo's existing renderer, see
app/agents/nodes/screenshot_utils.py) only exposes a `screenshot()` call —
it has no `page.evaluate()` equivalent to read `getComputedStyle`. Rather
than add a new browser-automation dependency, this module reuses the exact
same headless-Chrome executable and flags html2image drives (via
`make_hti`), and drives Chrome's own `--dump-dom` CLI mode instead of
`--screenshot`: a small vanilla `<script>` is injected into the rendered
HTML that tags every element carrying a direct text node with its
`getComputedStyle(el).fontSize` as a `data-font-px` attribute, and
`--dump-dom` prints the resulting (post-JS) DOM to stdout for parsing.
This is still "the existing rendering pipeline" — same browser binary,
same container-safety flags — just a different Chrome CLI mode.

Not every small font in this template is a bug: uppercase micro-labels
(section eyebrows, badges, table headers, the footer/progress row) are a
deliberate editorial-design pattern and are always < 14px by design. The
`_is_decorative` filter excludes those known chrome elements so the
assertion targets actual reading content (body copy, table data cells,
metric readouts) — which is what a real mobile legibility bug would shrink.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.request import pathname2url

import pytest
from bs4 import BeautifulSoup, Tag
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.agents.nodes.screenshot_utils import CHROME_FLAGS, make_hti

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_DIR = _REPO_ROOT / "app" / "templates"

_MOBILE_WIDTH = 350
_MOBILE_HEIGHT = 622
_MIN_CONTENT_FONT_PX = 14.0

# Uppercase micro-labels / chrome elements that are deliberately small by
# design (editorial small-caps pattern), not primary reading content.
_DECORATIVE_CLASSES = {
    "sys-label", "badge", "section-eyebrow", "progress-label",
    "footer-row", "winner-badge", "stat-label", "dim-name",
    "contrib-num", "url-chip", "cta-sub", "spec-tile-label",
}
_DECORATIVE_TAGS = {"th"}

_MEASURE_SCRIPT = """
<script>
(function () {
  var all = document.body.querySelectorAll('*');
  for (var i = 0; i < all.length; i++) {
    var el = all[i];
    var tag = el.tagName.toLowerCase();
    if (tag === 'script' || tag === 'style') { continue; }
    var hasDirectText = false;
    for (var j = 0; j < el.childNodes.length; j++) {
      var n = el.childNodes[j];
      if (n.nodeType === 3 && n.textContent.trim().length > 0) {
        hasDirectText = true;
        break;
      }
    }
    if (hasDirectText) {
      el.setAttribute('data-font-px', window.getComputedStyle(el).fontSize);
    }
  }
})();
</script>
"""


@dataclass(frozen=True)
class FontSample:
    """One visible text-bearing element's computed font size."""

    tag: Tag
    font_px: float
    text: str


def _chrome_executable() -> str:
    """Reuse the exact browser html2image would launch (see screenshot_utils.make_hti)."""
    hti = make_hti(_REPO_ROOT / "output" / "images", (1080, 1080))
    return str(hti.browser.executable)


def _skip_if_no_chrome() -> None:
    try:
        _chrome_executable()
    except FileNotFoundError:
        pytest.skip("No Chrome/Chromium executable available in this environment")


def _render_and_measure(
    slide_type: str, context: dict[str, object], tmp_path: Path
) -> list[FontSample]:
    """Render one slide at mobile width and return every visible text element's
    real computed font size, via headless Chrome's --dump-dom (see module docstring)."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("research_carousel_slide.html")
    html = template.render(slide_type=slide_type, **context)
    html = html.replace("</body>", _MEASURE_SCRIPT + "</body>")

    html_path = tmp_path / f"{slide_type}.html"
    html_path.write_text(html, encoding="utf-8")
    file_url = "file:" + pathname2url(str(html_path))

    command = [
        _chrome_executable(),
        "--headless=new",
        *CHROME_FLAGS,
        f"--window-size={_MOBILE_WIDTH},{_MOBILE_HEIGHT}",
        "--virtual-time-budget=4000",
        "--dump-dom",
        file_url,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Chrome exited {result.returncode} while measuring slide {slide_type!r}: "
            f"{result.stderr[:2000]}"
        )

    soup = BeautifulSoup(result.stdout, "html.parser")
    samples: list[FontSample] = []
    for el in soup.find_all(attrs={"data-font-px": True}):
        if el.name in ("script", "style"):
            continue
        font_px_str = el.get("data-font-px", "")
        if not font_px_str.endswith("px"):
            continue
        samples.append(
            FontSample(tag=el, font_px=float(font_px_str[:-2]), text=el.get_text(strip=True))
        )
    return samples


def _is_decorative(sample: FontSample) -> bool:
    """True if `sample` (or an ancestor) is a known small-by-design chrome element."""
    for el in (sample.tag, *sample.tag.parents):
        if not isinstance(el, Tag):
            continue
        if el.name in _DECORATIVE_TAGS:
            return True
        classes = el.get("class") or []
        if _DECORATIVE_CLASSES.intersection(classes):
            return True
        style = (el.get("style") or "").lower()
        if "letter-spacing" in style:
            return True
        # Unclassed one-off eyebrow/URL labels (e.g. "Full paper ->", "vs {prior paper}")
        # set JetBrains Mono directly inline rather than via a reusable class; in this
        # template mono + <=12px is always a chrome label, never reading content (the
        # mono *content* elements — stat-value, metric-text/-arrow — are all >=16px).
        if "jetbrains mono" in style and sample.font_px <= 12:
            return True
    return False


def _content_samples(samples: list[FontSample]) -> list[FontSample]:
    return [s for s in samples if not _is_decorative(s)]


def _min_content_font(samples: list[FontSample]) -> FontSample:
    content = _content_samples(samples)
    assert content, "No content (non-decorative) text elements were found to measure"
    return min(content, key=lambda s: s.font_px)


# ── Representative slide contexts (mirrors research_carousel_node's slide_defs) ──

_PRIOR_ART_CONTEXT = {
    "prior_art": {
        "prior_paper_name": "GPT-4 Baseline",
        "dimensions": [
            {"dimension": "Accuracy", "prior_sota": "85%", "new_paper": "92%", "winner": "new"},
            {"dimension": "Latency", "prior_sota": "120ms", "new_paper": "80ms", "winner": "new"},
            {"dimension": "Params", "prior_sota": "70B", "new_paper": "13B", "winner": "new"},
        ],
        "overall_verdict": "Clear improvement across every measured dimension.",
    },
    "technical_innovation": "A learned routing mechanism replaces fixed sparse attention windows.",
    "slide_num": 3,
    "total_slides": 10,
}

_RESULTS_CONTEXT = {
    "claims_evidence": [
        {"claim": "Outperforms GPT-4-Turbo on MMLU", "evidence_status": "quantified"},
        {"claim": "Faster inference at long context", "evidence_status": "quantified"},
        {"claim": "Generalizes to unseen domains", "evidence_status": "qualitative"},
    ],
    "quantitative_results": ["MMLU: 89.2% (+3.1%)", "HellaSwag: 95.1% (+1.8%)"],
    "breakthroughs": "Achieves state-of-the-art accuracy at a fraction of the inference cost.",
    "benchmark_chart_b64": "",
    "slide_num": 7,
    "total_slides": 10,
}

_SLIDE_CONTEXTS: dict[str, dict[str, object]] = {
    "cover": {
        "title": "Efficient Sparse Attention via Learned Token Routing",
        "hook": "A new routing mechanism cuts attention memory from O(n^2) to O(n log n).",
        "significance_verdict": "Major Contribution",
        "paper_url": "https://arxiv.org/abs/2401.99999",
        "is_classic_paper": False,
        "score_gauges_html": "",
        "slide_num": 1,
        "total_slides": 10,
    },
    "problem": {
        "core_problem": (
            "Existing attention is O(n^2) in memory, capping context length on consumer GPUs."
        ),
        "executive_summary_para1": (
            "Long documents must be chunked today, losing coherence across sections."
        ),
        "slide_num": 2,
        "total_slides": 10,
    },
    "prior_art": _PRIOR_ART_CONTEXT,
    "methodology": {
        "methodology": (
            "Sparse attention with learned per-token routing reduces complexity to O(n log n)."
        ),
        "technical_innovation": "Routing is dynamic per token rather than a fixed local window.",
        "architecture_diagram_b64": "",
        "architecture_spec_svg": "",
        "architecture_fallback_text": "",
        "slide_num": 4,
        "total_slides": 10,
    },
    "innovations": {
        "key_contributions": [
            "Sparse attention: O(n^2) to O(n log n) memory, proven on 3 architectures.",
            "Learned routing MLP trained end-to-end via straight-through estimators.",
        ],
        "methodology_fallback": "",
        "slide_num": 5,
        "total_slides": 10,
    },
    "experiments": {
        "experiment_spec": {
            "dataset": "C4 (350B tokens)",
            "baselines": "GPT-4-Turbo, Mistral-7B",
            "metrics": "MMLU, HellaSwag, ARC-Challenge",
            "compute": "256 A100s x 10 days",
        },
        "experiment_setup": "",
        "methodology_fallback": "",
        "slide_num": 6,
        "total_slides": 10,
    },
    "results": _RESULTS_CONTEXT,
    "impact": {
        "real_world_applications": [
            "Long-document summarization in legal tech.",
            "Code review assistants handling full-repository context.",
        ],
        "ecosystem_impact": "HuggingFace Transformers needs a sparse-attention backend.",
        "expert_interpretation": "",
        "slide_num": 9,
        "total_slides": 10,
    },
    "takeaways": {
        "future_directions": [
            "Extend routing to multi-modal inputs.",
            "Apply to diffusion transformers for image generation.",
        ],
        "limitations": "Only evaluated on English benchmarks so far.",
        "expert_interpretation": "",
        "paper_url": "https://arxiv.org/abs/2401.99999",
        "slide_num": 10,
        "total_slides": 10,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("slide_type", sorted(_SLIDE_CONTEXTS))
def test_content_meets_mobile_font_minimum(slide_type: str, tmp_path: Path) -> None:
    """Every core slide's reading content must render at >= 14px even at 350px width."""
    _skip_if_no_chrome()

    samples = _render_and_measure(slide_type, _SLIDE_CONTEXTS[slide_type], tmp_path)
    smallest = _min_content_font(samples)

    assert smallest.font_px >= _MIN_CONTENT_FONT_PX, (
        f"Slide {slide_type!r} has content text at {smallest.font_px}px "
        f"(text: {smallest.text[:60]!r}), below the {_MIN_CONTENT_FONT_PX}px mobile minimum"
    )


@pytest.mark.slow
def test_comparison_table_data_cells_meet_mobile_minimum(tmp_path: Path) -> None:
    """Reference failure case: the prior-art comparison table's data values
    (not its header labels) must stay legible at mobile width."""
    _skip_if_no_chrome()

    samples = _render_and_measure("prior_art", _PRIOR_ART_CONTEXT, tmp_path)
    data_cells = [
        s for s in samples
        if s.tag.name == "td" and "dim-name" not in (s.tag.get("class") or []) and s.text
    ]
    assert data_cells, "Expected at least one comparison-table data cell to measure"
    smallest = min(data_cells, key=lambda s: s.font_px)

    assert smallest.font_px >= _MIN_CONTENT_FONT_PX, (
        f"Comparison table data cell {smallest.text!r} renders at {smallest.font_px}px, "
        f"below the {_MIN_CONTENT_FONT_PX}px mobile minimum"
    )


@pytest.mark.slow
def test_results_metric_rows_meet_mobile_minimum(tmp_path: Path) -> None:
    """Results-slide metric readouts (the other half of the reference failure case)
    must also stay legible at mobile width.

    `.metric-row`/`.metric-text` only render in the template's plain-metrics-list
    branch (no `claims_evidence`), so this uses a dedicated context rather than
    `_RESULTS_CONTEXT` (which exercises the claims-vs-evidence branch instead)."""
    _skip_if_no_chrome()

    metrics_only_context = {
        **_RESULTS_CONTEXT,
        "claims_evidence": [],
        "quantitative_results": ["MMLU: 89.2% (+3.1%)", "HellaSwag: 95.1% (+1.8%)"],
    }
    samples = _render_and_measure("results", metrics_only_context, tmp_path)
    metric_texts = [s for s in samples if "metric-text" in (s.tag.get("class") or [])]
    assert metric_texts, "Expected at least one metric-text element to measure"
    smallest = min(metric_texts, key=lambda s: s.font_px)

    assert smallest.font_px >= _MIN_CONTENT_FONT_PX, (
        f"Results metric row {smallest.text!r} renders at {smallest.font_px}px, "
        f"below the {_MIN_CONTENT_FONT_PX}px mobile minimum"
    )


@pytest.mark.slow
def test_decorative_labels_are_intentionally_excluded(tmp_path: Path) -> None:
    """Sanity check that the decorative filter is doing something: the cover
    slide's system label really is < 14px, and really is filtered out."""
    _skip_if_no_chrome()

    samples = _render_and_measure("cover", _SLIDE_CONTEXTS["cover"], tmp_path)
    sys_labels = [s for s in samples if "sys-label" in (s.tag.get("class") or [])]
    assert sys_labels, "Expected the cover slide's sys-label element to be measured"
    assert sys_labels[0].font_px < _MIN_CONTENT_FONT_PX
    content = _content_samples(samples)
    assert all("sys-label" not in (s.tag.get("class") or []) for s in content)

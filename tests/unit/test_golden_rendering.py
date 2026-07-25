"""
Golden-file (snapshot) rendering tests for the research carousel template.

Renders each synthetic fixture in tests/golden/fixtures.py through the exact
same rendering pipeline `research_carousel_node` uses (Jinja2 template render
+ html2image headless Chrome screenshot — see app/agents/nodes/screenshot_utils.py)
and diffs the result against an approved baseline PNG in tests/golden/baselines/.

Why this exists: nothing previously rendered representative papers to PNG and
diffed against approved images, so regressions like LaTeX leaking into a title
or a fallback text block silently breaking only surfaced when a human looked
at the actual carousel. A pixel-tolerance diff (rather than exact equality)
absorbs font-antialiasing noise between runs on the same machine while still
catching real layout/content regressions.

No LLM or network call is made: fixtures are static dicts, and Chrome's
Google Fonts request is redirected to loopback (see `_EXTRA_CHROME_FLAGS`)
so font-fallback rendering is deterministic across runs regardless of
ambient network access.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageChops

from app.agents.nodes.screenshot_utils import CHROME_FLAGS, capture_slide
from tests.golden.fixtures import GOLDEN_FIXTURES, GoldenFixture

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_DIR = _REPO_ROOT / "app" / "templates"
_BASELINES_DIR = _REPO_ROOT / "tests" / "golden" / "baselines"

# Redirect the template's Google Fonts @import to loopback so a render never
# blocks on (or varies with) real network access — the font *family* doesn't
# affect font-*size* (covered separately in test_mobile_rendering.py), but an
# unreachable-vs-reachable CDN would otherwise make pixel diffs nondeterministic.
_EXTRA_CHROME_FLAGS = [
    "--host-resolver-rules=MAP fonts.googleapis.com 127.0.0.1,MAP fonts.gstatic.com 127.0.0.1",
]

# Fraction of pixels allowed to differ (by more than _CHANNEL_DELTA in any
# channel) before a diff is treated as a real regression rather than noise.
_DIFF_RATIO_TOLERANCE = 0.01
_CHANNEL_DELTA = 30


def _render_fixture(fixture: GoldenFixture, output_dir: Path) -> Path:
    """Render one fixture's slide through the real Jinja2 + html2image pipeline.

    Uses `capture_slide` (the same production helper `research_carousel_node`
    calls) rather than a bare `Html2Image.screenshot()`: html2image shells out
    to headless Chrome via `subprocess.run()` without checking the return
    code, so a renderer crash silently leaves an absent (or empty) PNG instead
    of raising — `capture_slide` retries once to absorb that transient
    failure mode. A retry also guards the empty-file case specifically,
    since `capture_slide`'s own existence check can pass on a 0-byte file
    written just before a crash.
    """
    from app.agents.nodes.screenshot_utils import make_hti

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("research_carousel_slide.html")
    html = template.render(slide_type=fixture.slide_type, **fixture.context)

    filename = f"{fixture.name}.png"
    hti = make_hti(output_dir, (1080, 1080))
    hti.browser.flags = [*CHROME_FLAGS, *_EXTRA_CHROME_FLAGS]

    max_attempts = 5
    rendered_path: Path | None = None
    for attempt in range(1, max_attempts + 1):
        result = capture_slide(hti, html, filename, label=fixture.name, output_dir=output_dir)
        candidate = Path(result) if result else None
        if candidate and candidate.exists() and candidate.stat().st_size > 0:
            rendered_path = candidate
            break
        # A full-suite run spawns many Chrome subprocesses back-to-back (the
        # mobile-rendering tests alone launch 12+ before this module runs),
        # and a crash under that contention tends to clear within a second —
        # a short backoff avoids re-hitting the same transient resource
        # pressure that produced the empty file in the first place.
        time.sleep(0.75 * attempt)
    if rendered_path is None:
        raise AssertionError(
            f"Chrome did not produce a valid (non-empty) screenshot for fixture "
            f"{fixture.name!r} after {max_attempts} attempts"
        )
    return rendered_path


def _diff_ratio(
    image_a: Image.Image, image_b: Image.Image, channel_delta: int = _CHANNEL_DELTA
) -> float:
    """Fraction of pixels whose max per-channel difference exceeds `channel_delta`."""
    if image_a.size != image_b.size:
        return 1.0
    diff = ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB"))
    r, g, b = diff.split()
    max_channel_diff = ImageChops.lighter(ImageChops.lighter(r, g), b)
    changed = max_channel_diff.point(lambda p: 255 if p > channel_delta else 0)
    histogram = changed.histogram()
    changed_pixels = histogram[255] if len(histogram) > 255 else 0
    total_pixels = changed.size[0] * changed.size[1]
    return changed_pixels / total_pixels


def _skip_if_no_chrome() -> None:
    try:
        from html2image import Html2Image

        Html2Image()
    except FileNotFoundError:
        pytest.skip("No Chrome/Chromium executable available in this environment")


@pytest.mark.slow
@pytest.mark.parametrize("fixture", GOLDEN_FIXTURES, ids=lambda f: f.name)
def test_slide_matches_golden_baseline(fixture: GoldenFixture, tmp_path: Path) -> None:
    """Each fixture's rendered slide must match its approved baseline within tolerance."""
    _skip_if_no_chrome()

    rendered_path = _render_fixture(fixture, tmp_path)
    baseline_path = _BASELINES_DIR / f"{fixture.name}.png"

    if not baseline_path.exists():
        _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        Image.open(rendered_path).convert("RGB").save(baseline_path)
        pytest.skip(
            f"No approved baseline existed for {fixture.name!r} — created one at "
            f"{baseline_path}. Re-run to verify against it."
        )

    rendered = Image.open(rendered_path).convert("RGB")
    baseline = Image.open(baseline_path).convert("RGB")

    ratio = _diff_ratio(rendered, baseline)
    assert ratio <= _DIFF_RATIO_TOLERANCE, (
        f"Rendered slide for fixture {fixture.name!r} differs from its approved baseline "
        f"by {ratio:.2%} of pixels (tolerance {_DIFF_RATIO_TOLERANCE:.2%}). "
        f"If this change is intentional, delete {baseline_path} and re-run to re-approve."
    )


@pytest.mark.slow
def test_all_fixtures_have_unique_names() -> None:
    """Guards against accidental fixture-name collisions clobbering each other's baseline."""
    names = [f.name for f in GOLDEN_FIXTURES]
    assert len(names) == len(set(names))


@pytest.mark.slow
def test_at_least_three_fixtures_defined() -> None:
    """Spec requires 3-4 representative synthetic paper fixtures."""
    assert 3 <= len(GOLDEN_FIXTURES) <= 4

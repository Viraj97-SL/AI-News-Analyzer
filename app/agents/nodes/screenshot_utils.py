"""
Shared headless-Chrome screenshot helpers for html2image-based renderers.

Centralises the browser flags and the file-existence check so all four
call sites (news carousel, research carousel, research card fallback,
prior-art card) stay in sync instead of drifting independently.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

CHROME_FLAGS = [
    "--no-sandbox",
    "--hide-scrollbars",
    "--disable-gpu",
    # Docker's default /dev/shm is 64MB; Chrome's renderer silently crashes
    # (no Python exception — html2image doesn't check the subprocess return
    # code) once a page's paint surface exceeds it. This forces Chrome to
    # spill to /tmp instead, which isn't quota-limited the same way.
    "--disable-dev-shm-usage",
]


def make_hti(output_dir: Path | str, size: tuple[int, int]):
    """Construct an Html2Image instance with the shared container-safe flags."""
    from html2image import Html2Image

    return Html2Image(
        output_path=str(output_dir),
        size=size,
        custom_flags=CHROME_FLAGS,
    )


def capture_slide(hti, html: str, filename: str, label: str, output_dir: Path | str) -> str | None:
    """
    Take one screenshot and verify the file actually landed on disk.

    html2image shells out to headless Chrome via `subprocess.run()` without
    checking the return code, so a renderer crash never raises a Python
    exception — the PNG is just silently absent. Retrying once catches
    transient crashes; the named warning/error means a persistent failure
    points at a specific slide instead of a bare count.

    `output_dir` is taken explicitly (the same directory passed to
    `make_hti`) rather than read back off `hti.output_path`, since callers
    may pass a mocked/stubbed `hti` in tests.
    """
    out_path = Path(output_dir) / filename
    for attempt in (1, 2):
        hti.screenshot(html_str=html, save_as=filename)
        if out_path.exists():
            return str(out_path)
        logger.warning("slide_capture_failed", slide=label, attempt=attempt)
    logger.error("slide_capture_gave_up", slide=label)
    return None

"""
Image generation node — news cards + LinkedIn carousel PDF via html2image.

Produces:
  - 1200x627px individual news cards (used in email newsletter)
  - 1080x1080px carousel slides combined into a PDF (used for LinkedIn document post)

Both use HTML/CSS templates rendered through headless Chrome. Zero API cost.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


def _extract_key_points(body: str) -> list[str]:
    """Split a summary body into 2–3 readable bullet points."""
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    return [s for s in sentences if len(s) > 15][:3]


def _make_hti(size: tuple[int, int]):
    """Create an Html2Image instance for the given canvas size."""
    from html2image import Html2Image

    return Html2Image(
        output_path=str(OUTPUT_DIR),
        size=size,
        custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
    )


def _generate_cards(summaries: list[dict], run_id: str, env: Environment) -> list[str]:
    """Render individual 1200×627 news cards (for email attachments)."""
    template = env.get_template("news_card.html")
    hti = _make_hti((1200, 627))
    paths: list[str] = []

    for i, summary in enumerate(summaries[:5]):
        html = template.render(
            headline=summary.get("headline", "AI News Update"),
            body=summary.get("body", "")[:180],
            category=summary.get("category", "AI"),
            source_count=len(summary.get("source_urls", [])),
            credibility=f"{summary.get('credibility_score', 0):.0%}",
            run_id=run_id,
        )
        filename = f"card_{run_id}_{i}.png"
        hti.screenshot(html_str=html, save_as=filename)
        paths.append(str(OUTPUT_DIR / filename))

    return paths


def _generate_carousel_pdf(summaries: list[dict], run_id: str, env: Environment) -> str | None:
    """
    Render 1080×1080 carousel slides and combine them into a single PDF.

    Slide order: cover → one story per summary → closing CTA.
    Returns the path to the PDF, or None on failure.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("pillow_not_installed", hint="pip install Pillow")
        return None

    template = env.get_template("carousel_slide.html")
    hti = _make_hti((1080, 1080))

    story_summaries = summaries[:7]
    total_slides = len(story_summaries)
    date_str = date.today().strftime("%B %d, %Y")
    slide_pngs: list[str] = []

    # ── Cover slide ──────────────────────────────────────
    cover_html = template.render(
        slide_type="cover",
        story_count=total_slides,
        date_str=date_str,
    )
    cover_name = f"carousel_{run_id}_cover.png"
    hti.screenshot(html_str=cover_html, save_as=cover_name)
    slide_pngs.append(str(OUTPUT_DIR / cover_name))

    # ── One story slide per summary ──────────────────────
    for i, summary in enumerate(story_summaries):
        key_points = _extract_key_points(summary.get("body", ""))
        story_html = template.render(
            slide_type="story",
            slide_num=i + 1,
            total_slides=total_slides,
            headline=summary.get("headline", ""),
            key_points=key_points,
            category=summary.get("category", "AI"),
            credibility=f"{summary.get('credibility_score', 0):.0%}",
        )
        name = f"carousel_{run_id}_{i}.png"
        hti.screenshot(html_str=story_html, save_as=name)
        slide_pngs.append(str(OUTPUT_DIR / name))

    # ── Closing / CTA slide ──────────────────────────────
    close_html = template.render(slide_type="closing")
    close_name = f"carousel_{run_id}_close.png"
    hti.screenshot(html_str=close_html, save_as=close_name)
    slide_pngs.append(str(OUTPUT_DIR / close_name))

    # ── Combine PNGs → PDF ───────────────────────────────
    pdf_path = str(OUTPUT_DIR / f"carousel_{run_id}.pdf")
    images = [Image.open(p).convert("RGB") for p in slide_pngs]
    images[0].save(pdf_path, save_all=True, append_images=images[1:])

    logger.info("carousel_generated", slides=len(images), pdf=pdf_path)
    return pdf_path


def image_gen_node(state: PipelineState) -> dict:
    """Generate news card images (email) and a carousel PDF (LinkedIn)."""
    summaries = state.get("summaries", [])
    if not summaries:
        logger.info("image_gen_skipped", reason="no summaries available")
        return {"image_paths": [], "current_step": "images_generated"}

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )

        run_id = state.get("run_id", "dev")

        # Individual cards for email
        image_paths = _generate_cards(summaries, run_id, env)
        logger.info("news_cards_generated", count=len(image_paths))

        # Carousel PDF for LinkedIn
        carousel_pdf = _generate_carousel_pdf(summaries, run_id, env)

        result: dict = {"image_paths": image_paths, "current_step": "images_generated"}
        if carousel_pdf:
            result["carousel_pdf_path"] = carousel_pdf

        return result

    except ImportError:
        logger.warning("html2image_not_installed", hint="pip install html2image")
        return {"image_paths": [], "error_log": ["html2image not installed"]}
    except Exception as e:
        logger.error("image_gen_error", error=str(e))
        return {"image_paths": [], "error_log": [f"Image gen error: {e}"]}

"""
Research carousel node — renders a 5-slide 1080×1080 PDF carousel for LinkedIn.

Slide order:
  1. Title + Hook
  2. Core Problem
  3. Methodology
  4. Benchmarks / Breakthroughs
  5. Takeaways + CTA
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


def research_carousel_node(state: "PipelineState") -> dict:
    """Render 5-slide 1080×1080 PNG slides and combine into a PDF for LinkedIn."""
    analysis = state.get("deep_analysis", {})
    paper = state.get("chosen_research_paper", {})
    linkedin_draft = state.get("linkedin_draft", "")
    run_id = state.get("run_id", "dev")

    if not analysis or not paper:
        return {
            "research_carousel_pdf_path": "",
            "research_carousel_slide_paths": [],
            "current_step": "research_carousel_skipped",
        }

    try:
        from html2image import Html2Image  # type: ignore[import]
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from PIL import Image  # type: ignore[import]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("research_carousel_slide.html")
        hti = Html2Image(
            output_path=str(OUTPUT_DIR),
            size=(1080, 1080),
            custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
        )

        # Hook = first 210 chars of linkedin_draft, split cleanly on word boundary
        hook = linkedin_draft[:210].rsplit(" ", 1)[0] if len(linkedin_draft) > 210 else linkedin_draft

        slide_defs = [
            {
                "slide_type": "title",
                "slide_num": 1,
                "title": paper.get("title", ""),
                "hook": hook,
                "paper_url": paper.get("url", ""),
            },
            {
                "slide_type": "problem",
                "slide_num": 2,
                "core_problem": analysis.get("core_problem", ""),
            },
            {
                "slide_type": "methodology",
                "slide_num": 3,
                "methodology": analysis.get("methodology", ""),
            },
            {
                "slide_type": "benchmarks",
                "slide_num": 4,
                "breakthroughs": analysis.get("breakthroughs", ""),
            },
            {
                "slide_type": "takeaways",
                "slide_num": 5,
                "limitations": analysis.get("limitations", ""),
                "paper_url": paper.get("url", ""),
            },
        ]

        slide_names = ["title", "problem", "methodology", "benchmarks", "takeaways"]
        slide_pngs: list[str] = []

        for slide_ctx, slide_name in zip(slide_defs, slide_names):
            html = template.render(**slide_ctx)
            filename = f"research_carousel_{run_id}_{slide_name}.png"
            hti.screenshot(html_str=html, save_as=filename)
            slide_pngs.append(str(OUTPUT_DIR / filename))

        existing = [p for p in slide_pngs if Path(p).exists()]
        if not existing:
            logger.error("research_carousel_no_slides_rendered")
            return {
                "research_carousel_pdf_path": "",
                "research_carousel_slide_paths": [],
                "current_step": "research_carousel_failed",
            }

        # Combine PNGs → PDF
        pdf_path = str(OUTPUT_DIR / f"research_carousel_{run_id}.pdf")
        images = [Image.open(p).convert("RGB") for p in existing]
        if len(images) == 1:
            images[0].save(pdf_path)
        else:
            images[0].save(pdf_path, save_all=True, append_images=images[1:])
        for img in images:
            img.close()

        logger.info("research_carousel_generated", slides=len(existing), pdf=pdf_path)
        return {
            "research_carousel_pdf_path": pdf_path,
            "research_carousel_slide_paths": existing,
            "current_step": "research_carousel_generated",
        }

    except Exception as e:
        logger.error("research_carousel_failed", error=str(e))
        return {
            "research_carousel_pdf_path": "",
            "research_carousel_slide_paths": [],
            "current_step": "research_carousel_failed",
        }

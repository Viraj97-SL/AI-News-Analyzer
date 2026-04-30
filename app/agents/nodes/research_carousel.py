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


_TOTAL_SLIDES = 10


def research_carousel_node(state: "PipelineState") -> dict:
    """Render 10-slide 1080×1080 PNG slides and combine into a PDF for LinkedIn."""
    analysis = state.get("deep_analysis", {})
    paper = state.get("chosen_research_paper", {})
    linkedin_draft = state.get("linkedin_draft", "")
    prior_art = state.get("prior_art_comparison", {})
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

        # Strip section-header lines (─── LABEL ───) produced by the LinkedIn prompt
        # before extracting the hook, so those labels never appear on the cover slide.
        clean_draft = "\n".join(
            line for line in linkedin_draft.splitlines()
            if not line.strip().startswith("───")
        ).strip()
        hook = clean_draft[:210].rsplit(" ", 1)[0] if len(clean_draft) > 210 else clean_draft

        slide_defs = [
            # 1 · Cover
            {
                "slide_type": "cover",
                "slide_num": 1,
                "total_slides": _TOTAL_SLIDES,
                "title": paper.get("title", ""),
                "hook": hook,
                "significance_verdict": analysis.get("significance_verdict", ""),
                "paper_url": paper.get("url", ""),
            },
            # 2 · The Problem
            {
                "slide_type": "problem",
                "slide_num": 2,
                "total_slides": _TOTAL_SLIDES,
                "core_problem": analysis.get("core_problem", ""),
                "executive_summary_para1": (analysis.get("executive_summary", "").split("\n\n") or [""])[0],
            },
            # 3 · Prior Art vs This Paper
            {
                "slide_type": "prior_art",
                "slide_num": 3,
                "total_slides": _TOTAL_SLIDES,
                "prior_art": prior_art,
                "technical_innovation": analysis.get("technical_innovation", ""),
            },
            # 4 · Methodology
            {
                "slide_type": "methodology",
                "slide_num": 4,
                "total_slides": _TOTAL_SLIDES,
                "methodology": analysis.get("methodology", ""),
                "technical_innovation": analysis.get("technical_innovation", ""),
            },
            # 5 · Key Innovations (numbered contributions)
            {
                "slide_type": "innovations",
                "slide_num": 5,
                "total_slides": _TOTAL_SLIDES,
                "key_contributions": analysis.get("key_contributions", []),
                "methodology_fallback": analysis.get("methodology", "")[:400],
            },
            # 6 · Experiments
            {
                "slide_type": "experiments",
                "slide_num": 6,
                "total_slides": _TOTAL_SLIDES,
                "experiment_setup": analysis.get("experiment_setup", ""),
                "methodology_fallback": analysis.get("methodology", "")[:300],
            },
            # 7 · Results
            {
                "slide_type": "results",
                "slide_num": 7,
                "total_slides": _TOTAL_SLIDES,
                "quantitative_results": analysis.get("quantitative_results", []),
                "breakthroughs": analysis.get("breakthroughs", ""),
            },
            # 8 · Ablation Study
            {
                "slide_type": "ablation",
                "slide_num": 8,
                "total_slides": _TOTAL_SLIDES,
                "ablation_highlights": analysis.get("ablation_highlights", ""),
                "limitations_fallback": analysis.get("limitations", "")[:350],
            },
            # 9 · Real-World Impact
            {
                "slide_type": "impact",
                "slide_num": 9,
                "total_slides": _TOTAL_SLIDES,
                "real_world_applications": analysis.get("real_world_applications", []),
                "ecosystem_impact": analysis.get("ecosystem_impact", ""),
                "expert_interpretation": analysis.get("expert_interpretation", ""),
            },
            # 10 · Takeaways + CTA
            {
                "slide_type": "takeaways",
                "slide_num": 10,
                "total_slides": _TOTAL_SLIDES,
                "future_directions": analysis.get("future_directions", []),
                "limitations": analysis.get("limitations", ""),
                "expert_interpretation": analysis.get("expert_interpretation", ""),
                "paper_url": paper.get("url", ""),
            },
        ]

        slide_names = [
            "cover", "problem", "prior_art", "methodology", "innovations",
            "experiments", "results", "ablation", "impact", "takeaways",
        ]
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

        # Use PyMuPDF (fitz) — bundled codecs, no libjpeg/openjpeg dependency.
        import fitz  # type: ignore[import]

        pdf_path = str(OUTPUT_DIR / f"research_carousel_{run_id}.pdf")
        doc = fitz.open()
        for png_path in existing:
            img_doc = fitz.open(png_path)
            pdf_bytes = img_doc.convert_to_pdf()
            img_doc.close()
            img_pdf = fitz.open("pdf", pdf_bytes)
            doc.insert_pdf(img_pdf)
            img_pdf.close()
        doc.save(pdf_path)
        doc.close()

        logger.info("research_carousel_generated", slides=len(existing), total=_TOTAL_SLIDES, pdf=pdf_path)
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

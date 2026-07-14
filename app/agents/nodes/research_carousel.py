"""
Research carousel node — renders a 1080×1080 PDF carousel for LinkedIn.

Base slide order (always present):
  1. Cover (title + hook + score gauges if scored)
  2. The Problem
  3. Prior Art vs This Paper
  4. Methodology (+ architecture diagram or ASCII fallback)
  5. Key Innovations
  6. Experiment Setup
  7. Results (+ benchmark chart if available)
  8. Ablation Study
  9. Real-World Impact
  10. Takeaways + CTA

An optional "Figures From The Paper" slide is inserted after Methodology
when 2+ figures were extracted (see `paper_figures` in PipelineState).
Slide numbering and the footer's "N / total" label are derived from the
final assembled slide list, not a hardcoded count.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.svg_gauge import render_gauge_svg
from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


_TOTAL_SLIDES = 10  # base slide count, excluding the optional figures slide


def research_carousel_node(state: "PipelineState") -> dict:
    """Render 10-slide 1080×1080 PNG slides and combine into a PDF for LinkedIn."""
    analysis = state.get("deep_analysis", {})
    paper = state.get("chosen_research_paper", {})
    linkedin_draft = state.get("linkedin_draft", "")
    prior_art = state.get("prior_art_comparison", {})
    run_id = state.get("run_id", "dev")
    is_classic = state.get("is_classic_paper", False)

    # Base64-encode the architecture diagram (already stored as b64 in state).
    # Treat a blank/whitespace-only value as absent so the fallback branch runs.
    arch_b64 = (state.get("architecture_diagram_b64", "") or "").strip()
    arch_fallback_text = state.get("architecture_fallback_text", "")
    paper_figures = state.get("paper_figures", []) or []

    research_scores = state.get("research_scores", {}) or {}
    score_gauges_html = ""
    if research_scores:
        gauge_defs = [
            ("Novelty", research_scores.get("novelty", 0), "#0EA5E9"),
            ("Clarity", research_scores.get("methodology_clarity", 0), "#7C3AED"),
            ("Benchmarks", research_scores.get("benchmark_improvement", 0), "#059669"),
            ("Repro", research_scores.get("reproducibility", 0), "#E11D48"),
        ]
        score_gauges_html = "".join(
            render_gauge_svg(label, value, color) for label, value, color in gauge_defs
        )

    # Base64-encode the benchmark chart PNG if available
    benchmark_chart_path = state.get("benchmark_chart_path", "")
    benchmark_chart_b64 = ""
    if benchmark_chart_path and Path(benchmark_chart_path).exists():
        try:
            benchmark_chart_b64 = base64.b64encode(
                Path(benchmark_chart_path).read_bytes()
            ).decode()

        except Exception:
            pass  # chart unavailable; slides render without it

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
            custom_flags=[
                "--no-sandbox",
                "--hide-scrollbars",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ],
        )

        # Strip section-header lines (─── LABEL ───) produced by the LinkedIn prompt
        # before extracting the hook, so those labels never appear on the cover slide.
        clean_draft = "\n".join(
            line for line in linkedin_draft.splitlines()
            if not line.strip().startswith("───")
        ).strip()
        hook = clean_draft[:210].rsplit(" ", 1)[0] if len(clean_draft) > 210 else clean_draft

        # slide_num / total_slides are assigned after assembly (see below) so the
        # optional figures slide can shift numbering without hardcoding a count.
        slide_defs: list[dict[str, object]] = [
            # Cover
            {
                "slide_type": "cover",
                "title": paper.get("title", ""),
                "hook": hook,
                "significance_verdict": analysis.get("significance_verdict", ""),
                "paper_url": paper.get("url", ""),
                "is_classic_paper": is_classic,
                "score_gauges_html": score_gauges_html,
            },
            # The Problem
            {
                "slide_type": "problem",
                "core_problem": analysis.get("core_problem", ""),
                "executive_summary_para1": (analysis.get("executive_summary", "").split("\n\n") or [""])[0],
            },
            # Prior Art vs This Paper
            {
                "slide_type": "prior_art",
                "prior_art": prior_art,
                "technical_innovation": analysis.get("technical_innovation", ""),
            },
            # Methodology — injects architecture diagram or ASCII fallback
            {
                "slide_type": "methodology",
                "methodology": analysis.get("methodology", ""),
                "technical_innovation": analysis.get("technical_innovation", ""),
                "architecture_diagram_b64": arch_b64,
                "architecture_fallback_text": arch_fallback_text,
            },
        ]

        # Optional: Figures From The Paper — only when 2+ real figures exist.
        if len(paper_figures) >= 2:
            slide_defs.append({
                "slide_type": "figures",
                "paper_figures": paper_figures[:4],
            })

        slide_defs.extend([
            # Key Innovations (numbered contributions)
            {
                "slide_type": "innovations",
                "key_contributions": analysis.get("key_contributions", []),
                "methodology_fallback": analysis.get("methodology", "")[:400],
            },
            # Experiments
            {
                "slide_type": "experiments",
                "experiment_setup": analysis.get("experiment_setup", ""),
                "methodology_fallback": analysis.get("methodology", "")[:300],
            },
            # Results — injects benchmark chart when available
            {
                "slide_type": "results",
                "quantitative_results": analysis.get("quantitative_results", []),
                "breakthroughs": analysis.get("breakthroughs", ""),
                "benchmark_chart_b64": benchmark_chart_b64,
            },
            # Ablation Study
            {
                "slide_type": "ablation",
                "ablation_highlights": analysis.get("ablation_highlights", ""),
                "limitations_fallback": analysis.get("limitations", "")[:350],
            },
            # Real-World Impact
            {
                "slide_type": "impact",
                "real_world_applications": analysis.get("real_world_applications", []),
                "ecosystem_impact": analysis.get("ecosystem_impact", ""),
                "expert_interpretation": analysis.get("expert_interpretation", ""),
            },
            # Takeaways + CTA
            {
                "slide_type": "takeaways",
                "future_directions": analysis.get("future_directions", []),
                "limitations": analysis.get("limitations", ""),
                "expert_interpretation": analysis.get("expert_interpretation", ""),
                "paper_url": paper.get("url", ""),
            },
        ])

        total_slides = len(slide_defs)
        for i, slide_ctx in enumerate(slide_defs, start=1):
            slide_ctx["slide_num"] = i
            slide_ctx["total_slides"] = total_slides

        slide_pngs: list[str] = []

        for slide_ctx in slide_defs:
            slide_name = slide_ctx["slide_type"]
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

        logger.info("research_carousel_generated", slides=len(existing), total=total_slides, pdf=pdf_path)
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

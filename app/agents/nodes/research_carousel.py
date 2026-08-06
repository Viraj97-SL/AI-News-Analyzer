"""
Research carousel node — renders a variable-length 1080×1080 PDF carousel
for LinkedIn. Slide count is not hardcoded: some slides only appear when
there's real structured content behind them, so a paper with no ablations
and no hero number produces a shorter deck instead of padded-out slides.

Core slide order (always present):
  1. Cover (title + hook + score gauges if scored)
  2. The Problem
  3. Prior Art vs This Paper
  4. Methodology (+ extracted figure, generated diagram, or ASCII fallback)
  5. Key Innovations
  6. Experiment Setup (spec-card grid when structured extraction succeeded)
  7. Results (claims-vs-evidence table + benchmark chart if available)
  8. Real-World Impact
  9. Takeaways + CTA

Conditional slides:
  - "Hook" (single hero number) — inserted right after Cover when
    `hook_stat` extraction succeeded.
  - "Figures From The Paper" — inserted after Methodology when 2+ figures
    were extracted (see `paper_figures` in PipelineState).
  - "Ablation Study" — included only when the paper has reported ablations
    or an extracted component breakdown; skipped entirely otherwise rather
    than rendering "no ablations mentioned".

Slide numbering and the footer's "N / total" label are derived from the
final assembled slide list, not a hardcoded count.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.svg_gauge import ACCENT_CONTRIBUTION, render_bar_strip_svg
from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


_TOTAL_SLIDES = 10  # base slide count, excluding the optional figures slide


def _count_dimension_winners(dimensions: list[dict]) -> dict[str, int]:
    """Tally winner counts across prior-art comparison dimensions, for the
    win-distribution bar on the Prior Art slide — a genuine summary of the
    already-extracted per-dimension verdicts, not a fabricated metric."""
    counts = {"new": 0, "prior": 0, "tie": 0}
    for dim in dimensions:
        winner = dim.get("winner") if isinstance(dim, dict) else getattr(dim, "winner", None)
        if winner in counts:
            counts[winner] += 1
    return counts


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
    arch_spec_svg = (state.get("architecture_spec_svg", "") or "").strip()
    paper_figures = state.get("paper_figures", []) or []

    experiment_spec = state.get("experiment_spec", {}) or {}
    claims_evidence = state.get("claims_evidence", []) or []
    ablation_components = state.get("ablation_components", []) or []
    ablation_chips_html = state.get("ablation_chips_html", "") or ""
    hook_stat = state.get("hook_stat", {}) or {}

    research_scores = state.get("research_scores", {}) or {}
    score_gauges_html = ""
    if research_scores:
        # All 4 scores describe this paper's own quantified/verified standing, so
        # they share the single "contribution" accent colour rather than one hue
        # per metric — colour here encodes ROLE (this paper's own numbers), not slide-type.
        gauge_defs = [
            ("Novelty", research_scores.get("novelty", 0), ACCENT_CONTRIBUTION),
            ("Clarity", research_scores.get("methodology_clarity", 0), ACCENT_CONTRIBUTION),
            ("Benchmarks", research_scores.get("benchmark_improvement", 0), ACCENT_CONTRIBUTION),
            ("Repro", research_scores.get("reproducibility", 0), ACCENT_CONTRIBUTION),
        ]
        score_gauges_html = "".join(
            render_bar_strip_svg(label, value, color) for label, value, color in gauge_defs
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
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        from app.agents.nodes.screenshot_utils import capture_slide, make_hti

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("research_carousel_slide.html")
        hti = make_hti(OUTPUT_DIR, (1080, 1080))

        # Strip section-header lines (─── LABEL ───) produced by the LinkedIn prompt
        # before extracting the hook, so those labels never appear on the cover slide.
        clean_draft = "\n".join(
            line for line in linkedin_draft.splitlines()
            if not line.strip().startswith("───")
        ).strip()
        hook = clean_draft[:210].rsplit(" ", 1)[0] if len(clean_draft) > 210 else clean_draft

        # slide_num / total_slides are assigned after assembly (see below) so
        # optional/skippable slides can shift numbering without hardcoding a count.
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
        ]

        # Optional: Hook — the single most surprising number, alone on its own slide.
        if hook_stat.get("value"):
            slide_defs.append({
                "slide_type": "hook_stat",
                "hook_stat_value": hook_stat.get("value", ""),
                "hook_stat_label": hook_stat.get("label", ""),
            })

        slide_defs.extend([
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
                "win_counts": _count_dimension_winners(prior_art.get("dimensions", [])),
            },
            # Methodology — injects extracted figure, generated diagram, or ASCII fallback
            {
                "slide_type": "methodology",
                "methodology": analysis.get("methodology", ""),
                "technical_innovation": analysis.get("technical_innovation", ""),
                "architecture_diagram_b64": arch_b64,
                "architecture_spec_svg": arch_spec_svg,
                "architecture_fallback_text": arch_fallback_text,
            },
        ])

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
            # Experiments — spec-card grid when structured extraction succeeded
            {
                "slide_type": "experiments",
                "experiment_spec": experiment_spec,
                "experiment_setup": analysis.get("experiment_setup", ""),
                "methodology_fallback": analysis.get("methodology", "")[:300],
            },
            # Results — claims-vs-evidence table + benchmark chart when available
            {
                "slide_type": "results",
                "claims_evidence": claims_evidence,
                "quantitative_results": analysis.get("quantitative_results", []),
                "breakthroughs": analysis.get("breakthroughs", ""),
                "benchmark_chart_b64": benchmark_chart_b64,
            },
        ])

        # Ablation Study — variable slide count: skip entirely when the paper has
        # neither reported ablations nor an extracted component breakdown, rather
        # than padding out a slide with "no ablations mentioned".
        ablation_text = analysis.get("ablation_highlights", "")
        has_ablation_content = bool(ablation_components) or (
            bool(ablation_text) and "no ablation" not in ablation_text.lower()
        )
        if has_ablation_content:
            slide_defs.append({
                "slide_type": "ablation",
                "ablation_chips_html": ablation_chips_html,
                "ablation_highlights": ablation_text,
                "limitations_fallback": analysis.get("limitations", "")[:350],
            })

        # Last-resort fallback text for slides whose preferred fields can come back
        # empty from the LLM (real_world_applications / future_directions are
        # optional lists) — core_problem and methodology are required fields on
        # RichDeepAnalysis, so they're the most reliable non-empty text available.
        summary_paragraphs = [p for p in analysis.get("executive_summary", "").split("\n\n") if p.strip()]
        fallback_text = (
            (summary_paragraphs[-1] if summary_paragraphs else "")
            or analysis.get("core_problem", "")
            or analysis.get("methodology", "")
        )[:450]

        slide_defs.extend([
            # Real-World Impact
            {
                "slide_type": "impact",
                "real_world_applications": analysis.get("real_world_applications", []),
                "ecosystem_impact": analysis.get("ecosystem_impact", ""),
                "expert_interpretation": analysis.get("expert_interpretation", ""),
                "impact_fallback": fallback_text,
            },
            # Takeaways + CTA
            {
                "slide_type": "takeaways",
                "future_directions": analysis.get("future_directions", []),
                "limitations": analysis.get("limitations", ""),
                "expert_interpretation": analysis.get("expert_interpretation", ""),
                "takeaways_fallback": fallback_text,
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
            path = capture_slide(hti, html, filename, label=slide_name, output_dir=OUTPUT_DIR)
            if path:
                slide_pngs.append(path)

        existing = slide_pngs
        if len(existing) < total_slides:
            logger.warning(
                "research_carousel_slides_missing",
                missing=total_slides - len(existing),
                total=total_slides,
            )
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

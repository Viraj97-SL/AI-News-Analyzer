"""
Prior art comparison node — extracts a structured vs-prior-SOTA comparison
and renders a 1200×627 cyberpunk comparison card.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


class ComparisonDimension(BaseModel):
    dimension: str = Field(description="e.g. 'Training Efficiency', 'MMLU Score', 'Parameter Count'")
    new_paper: str = Field(description="This paper's value or short description")
    prior_sota: str = Field(description="Prior SOTA value or short description")
    winner: Literal["new", "prior", "tie"] = Field(
        description="Which approach is better on this dimension"
    )


class PriorArtComparison(BaseModel):
    prior_paper_name: str = Field(description="Name of the main competing method or paper")
    dimensions: list[ComparisonDimension] = Field(description="3–5 comparison dimensions")
    overall_verdict: str = Field(description="One sentence summarising the net advancement")


def prior_art_node(state: "PipelineState") -> dict:
    """Extract prior-art comparison via LLM and render a comparison card PNG."""
    paper = state.get("chosen_research_paper", {})
    analysis = state.get("deep_analysis", {})
    run_id = state.get("run_id", "dev")

    if not paper or not analysis:
        return {
            "prior_art_comparison": {},
            "comparison_card_path": "",
            "current_step": "prior_art_skipped",
        }

    # ── 1. Structured extraction ──────────────────────────────────────────
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.3,
            api_key=settings.google_api_key,
        ).with_structured_output(PriorArtComparison)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an AI research expert. Compare this paper against the most relevant prior SOTA method. "
             "Identify 3–5 concrete dimensions where they differ (e.g. accuracy, efficiency, novelty, scalability). "
             "Be specific and quantitative where possible — only use values actually mentioned in the text."),
            ("user",
             "Title: {title}\n\n"
             "Core Problem: {core_problem}\n\n"
             "Methodology: {methodology}\n\n"
             "Breakthroughs: {breakthroughs}"),
        ])

        comparison: PriorArtComparison = (prompt | llm).invoke({
            "title": paper.get("title", ""),
            "core_problem": analysis.get("core_problem", ""),
            "methodology": analysis.get("methodology", ""),
            "breakthroughs": analysis.get("breakthroughs", ""),
        })
    except Exception as e:
        logger.error("prior_art_extraction_failed", error=str(e))
        return {
            "prior_art_comparison": {},
            "comparison_card_path": "",
            "current_step": "prior_art_skipped",
        }

    # ── 2. Render comparison card ─────────────────────────────────────────
    try:
        from html2image import Html2Image  # type: ignore[import]
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("prior_art_card.html")
        html = template.render(
            paper_title=paper.get("title", "Research Paper"),
            comparison=comparison.model_dump(),
        )

        hti = Html2Image(
            output_path=str(OUTPUT_DIR),
            size=(1200, 627),
            custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
        )
        filename = f"prior_art_card_{run_id}.png"
        hti.screenshot(html_str=html, save_as=filename)
        card_path = str(OUTPUT_DIR / filename)

        logger.info("prior_art_card_generated", path=card_path)
        return {
            "prior_art_comparison": comparison.model_dump(),
            "comparison_card_path": card_path,
            "current_step": "prior_art_card_generated",
        }
    except Exception as e:
        logger.error("prior_art_card_render_failed", error=str(e))
        return {
            "prior_art_comparison": comparison.model_dump(),
            "comparison_card_path": "",
            "current_step": "prior_art_extracted",
        }

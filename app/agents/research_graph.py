from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.nodes.approval import human_approval_node
from app.agents.nodes.architecture_diagram import architecture_diagram_node
from app.agents.nodes.benchmark_chart import benchmark_chart_node
from app.agents.nodes.prior_art import prior_art_node
from app.agents.nodes.research_carousel import research_carousel_node
from app.agents.nodes.scraper import scrape_arxiv_node
from app.agents.state import PipelineState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class PaperSelection(BaseModel):
    chosen_url: str = Field(description="The exact URL of the chosen paper.")
    reasoning: str = Field(description="1-sentence reason why this is the most impactful.")


class DeepAnalysis(BaseModel):
    core_problem: str = Field(description="What specific gap or problem is this paper solving?")
    methodology: str = Field(description="How did they solve it? Detail the architecture or math.")
    breakthroughs: str = Field(description="What were the quantifiable results or benchmarks?")
    limitations: str = Field(description="What are the drawbacks or future work needed?")


class ResearchScores(BaseModel):
    """Feature 8: 1-10 dimension scores for the research card gauges."""
    novelty: int = Field(description="Novelty score 1-10: how new is the core approach?", ge=1, le=10)
    methodology_clarity: int = Field(description="Methodology clarity score 1-10", ge=1, le=10)
    benchmark_improvement: int = Field(
        description="Benchmark improvement score 1-10: magnitude and breadth of gains", ge=1, le=10
    )
    reproducibility: int = Field(
        description="Reproducibility score 1-10: are code, data, and weights released?", ge=1, le=10
    )
    score_reasoning: str = Field(description="2-sentence justification for these scores.")


class HookScore(BaseModel):
    """Feature 1: Quality scores for the LinkedIn hook (first 210 chars)."""
    curiosity: int = Field(description="Curiosity gap score 0-10", ge=0, le=10)
    specificity: int = Field(description="Specificity score 0-10", ge=0, le=10)
    controversy: int = Field(description="Controversy or boldness score 0-10", ge=0, le=10)
    reasoning: str = Field(description="One sentence identifying the weakest dimension.")


# ── SVG Gauge Helper (Feature 8) ─────────────────────────────────────────────

def _render_gauge_svg(label: str, value: int, color: str) -> str:
    """Return an inline SVG radial gauge with neon glow for a 1-10 score."""
    r = 40.0
    circumference = 2 * math.pi * r   # 251.33
    arc_span = circumference * 0.75    # 188.50 (270°)
    fill_len = (value / 10) * arc_span
    gap = circumference - arc_span     # 62.83

    bg_dash = f"{arc_span:.2f} {gap:.2f}"
    fg_dash = f"{fill_len:.2f} {circumference - fill_len:.2f}"
    transform = "rotate(135 60 60)"

    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px">'
        f'<svg width="100" height="100" viewBox="0 0 120 120">'
        f'<circle cx="60" cy="60" r="{r:.0f}" fill="none"'
        f' stroke="rgba(255,255,255,0.08)" stroke-width="9" stroke-linecap="round"'
        f' stroke-dasharray="{bg_dash}" transform="{transform}"/>'
        f'<circle cx="60" cy="60" r="{r:.0f}" fill="none"'
        f' stroke="{color}" stroke-width="9" stroke-linecap="round"'
        f' stroke-dasharray="{fg_dash}" transform="{transform}"'
        f' style="filter:drop-shadow(0 0 5px {color})"/>'
        f'<text x="60" y="55" text-anchor="middle" fill="white"'
        f' font-size="24" font-weight="700" font-family="JetBrains Mono,monospace">{value}</text>'
        f'<text x="60" y="72" text-anchor="middle" fill="rgba(255,255,255,0.35)"'
        f' font-size="11" font-family="JetBrains Mono,monospace">/10</text>'
        f'</svg>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        f'color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:1.5px;'
        f'text-align:center;margin-top:-4px">{label}</div>'
        f'</div>'
    )


# ── 1. Intelligence Nodes ────────────────────────────────────────────────────

def select_paper_node(state: PipelineState) -> dict:
    """Uses Gemini Flash to select the single most impactful ArXiv paper."""
    logger.info("research_node_running", step="selecting_best_paper")
    articles = state.get("raw_articles", [])

    if not articles:
        return {"current_step": "no_papers_found"}

    papers_text = "\n\n".join([
        f"URL: {a['url']}\nTitle: {a['title']}\nAbstract: {a['content'][:1000]}"
        for a in articles[:30]
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=settings.google_api_key,
    ).with_structured_output(PaperSelection)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Principal Investigator of an elite AI lab. Read these recent ArXiv paper abstracts. "
         "Select the SINGLE most groundbreaking, highly-novel paper that engineers and researchers absolutely "
         "must know about. Favor novel architectures and major benchmark breakthroughs over minor optimizations."),
        ("user", "{papers}"),
    ])

    result = (prompt | llm).invoke({"papers": papers_text})
    chosen_paper = next((a for a in articles if a["url"] == result.chosen_url), articles[0])

    logger.info("paper_selected", title=chosen_paper["title"])
    return {"chosen_research_paper": chosen_paper, "current_step": "paper_selected"}


def deep_analysis_node(state: PipelineState) -> dict:
    """Uses Gemini Pro to extract technical analysis and draft algorithm-optimised LinkedIn post."""
    logger.info("research_node_running", step="deep_analysis")
    paper = state.get("chosen_research_paper")

    if not paper:
        return {"current_step": "error_no_paper"}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.4,
        api_key=settings.google_api_key,
    ).with_structured_output(DeepAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Senior AI Research Scientist. Analyse this paper. Extract deep technical concepts, "
         "explaining them clearly for a highly technical audience (PhDs, ML Engineers). "
         "Do not use fluff or generic marketing speak."),
        ("user", "Title: {title}\n\nContent: {content}"),
    ])

    analysis: DeepAnalysis = (prompt | llm).invoke({
        "title": paper["title"],
        "content": paper["content"],
    })

    # ── LinkedIn draft: hook-first, algorithm-optimised format ──
    core_snippet = analysis.core_problem[:180].rsplit(".", 1)[0]
    linkedin_draft = (
        f"🚨 This paper just changed how I think about AI: {paper['title'][:80]}\n\n"
        f"The problem nobody is solving:\n"
        f"{core_snippet}.\n\n"
        f"Here's what they did differently:\n\n"
        f"1/ {analysis.methodology[:200].rsplit(' ', 1)[0]}...\n\n"
        f"2/ Results that matter:\n"
        f"{analysis.breakthroughs[:200].rsplit(' ', 1)[0]}...\n\n"
        f"3/ Honest limitations:\n"
        f"{analysis.limitations[:150].rsplit(' ', 1)[0]}...\n\n"
        f"What does this mean for your work? Drop your take below 👇\n\n"
        f"🔗 Full paper in the first comment.\n\n"
        f"#AIResearch #MachineLearning #DeepLearning #LLM #ArXiv"
    )

    newsletter_html = f"""
    <div style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
        <h2 style="color: #0a66c2;">Deep Dive: {paper['title']}</h2>
        <p><strong>Read the paper:</strong> <a href="{paper['url']}">{paper['url']}</a></p>
        <h3 style="border-bottom: 1px solid #eee; padding-bottom: 5px;">The Core Problem</h3>
        <p>{analysis.core_problem}</p>
        <h3 style="border-bottom: 1px solid #eee; padding-bottom: 5px;">Innovative Methodology</h3>
        <p>{analysis.methodology}</p>
        <h3 style="border-bottom: 1px solid #eee; padding-bottom: 5px;">Key Breakthroughs</h3>
        <p>{analysis.breakthroughs}</p>
        <h3 style="border-bottom: 1px solid #eee; padding-bottom: 5px;">Limitations & Future Work</h3>
        <p>{analysis.limitations}</p>
    </div>
    """

    return {
        "deep_analysis": analysis.model_dump(),
        "linkedin_draft": linkedin_draft,
        "newsletter_html": newsletter_html,
        "current_step": "analysis_complete",
    }


def score_research_node(state: PipelineState) -> dict:
    """Feature 8: Score the paper on 4 dimensions for gauge widgets on the research card."""
    logger.info("research_node_running", step="scoring_research")
    paper = state.get("chosen_research_paper", {})
    analysis = state.get("deep_analysis", {})

    if not analysis:
        return {"research_scores": {}, "current_step": "research_scores_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            api_key=settings.google_api_key,
        ).with_structured_output(ResearchScores)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Score this AI research paper on 4 dimensions (each 1-10). Be critical and calibrated — "
             "a 10 in novelty means a paradigm shift like Transformers. A 7 means meaningfully new. "
             "For reproducibility: 10 = code + data + weights released; 5 = partial; 1 = nothing released."),
            ("user",
             "Title: {title}\n\nCore Problem: {core_problem}\n\n"
             "Methodology: {methodology}\n\nBreakthroughs: {breakthroughs}\n\nLimitations: {limitations}"),
        ])

        scores: ResearchScores = (prompt | llm).invoke({
            "title": paper.get("title", ""),
            "core_problem": analysis.get("core_problem", ""),
            "methodology": analysis.get("methodology", ""),
            "breakthroughs": analysis.get("breakthroughs", ""),
            "limitations": analysis.get("limitations", ""),
        })

        logger.info(
            "research_scored",
            novelty=scores.novelty,
            clarity=scores.methodology_clarity,
            benchmarks=scores.benchmark_improvement,
            repro=scores.reproducibility,
        )
        return {"research_scores": scores.model_dump(), "current_step": "research_scored"}

    except Exception as e:
        logger.warning("research_scoring_failed", error=str(e))
        return {"research_scores": {}, "current_step": "research_scores_skipped"}


def score_hook_node(state: PipelineState) -> dict:
    """Feature 1: Score LinkedIn hook quality; regenerate up to 3× if score < 21/30."""
    logger.info("research_node_running", step="scoring_hook")
    linkedin_draft = state.get("linkedin_draft", "")
    attempts = state.get("hook_attempts", 0)

    if not linkedin_draft:
        return {"hook_score": {}, "current_step": "hook_skipped"}

    flash = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=settings.google_api_key,
    )

    # Up to 3 scoring + rewrite passes inline (keeps graph DAG simple)
    for _pass in range(3):
        hook = linkedin_draft[:210].rsplit(" ", 1)[0] if len(linkedin_draft) > 210 else linkedin_draft

        try:
            score_llm = flash.with_structured_output(HookScore)
            score_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a LinkedIn growth expert. Score this post hook on three axes, each 0-10:\n"
                 "• curiosity_gap: Does it create an open loop the reader must close?\n"
                 "• specificity: Does it use concrete numbers, names, or claims — not vague?\n"
                 "• controversy: Is it bold enough to stop the scroll?\n"
                 "Be strict. A generic 'AI is amazing' hook scores 2-3, not 8."),
                ("user", "HOOK (first 210 chars): {hook}"),
            ])
            score: HookScore = (score_prompt | score_llm).invoke({"hook": hook})
            total = score.curiosity + score.specificity + score.controversy

            logger.info("hook_scored", total=total, attempts=attempts + _pass)

            if total >= 21 or (attempts + _pass) >= 2:
                return {
                    "hook_score": score.model_dump() | {"total": total},
                    "hook_attempts": attempts + _pass,
                    "linkedin_draft": linkedin_draft,
                    "current_step": "hook_approved",
                }

            # Rewrite the hook
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Rewrite ONLY the first 210 characters of this LinkedIn post to score higher. "
                 "Weakness to fix: {weakness}\n"
                 "Rules: ≤210 chars, no external links, must be a single punchy statement or question. "
                 "Return ONLY the new hook text, nothing else."),
                ("user", "Current hook: {hook}\n\nFull post for context:\n{full_post}"),
            ])
            new_hook = (rewrite_prompt | flash).invoke({
                "weakness": score.reasoning,
                "hook": hook,
                "full_post": linkedin_draft[:600],
            }).content.strip()

            # Splice the new hook back in: replace content up to first double-newline
            rest_start = linkedin_draft.find("\n\n")
            if rest_start != -1:
                linkedin_draft = new_hook + linkedin_draft[rest_start:]
            else:
                linkedin_draft = new_hook

        except Exception as e:
            logger.warning("hook_scoring_pass_failed", error=str(e))
            break

    return {
        "hook_score": {},
        "hook_attempts": attempts,
        "linkedin_draft": linkedin_draft,
        "current_step": "hook_approved",
    }


def paperbanana_visual_node(state: PipelineState) -> dict:
    """Render the enhanced research card with gauges, arch diagram, and benchmark chart."""
    logger.info("research_node_running", step="generating_visual")
    analysis = state.get("deep_analysis", {})
    paper = state.get("chosen_research_paper", {})
    run_id = state.get("run_id", "test")
    research_scores = state.get("research_scores", {})
    benchmark_chart_path = state.get("benchmark_chart_path", "")
    arch_b64 = state.get("architecture_diagram_b64", "")
    arch_fallback = state.get("architecture_fallback_text", "")

    image_paths: list[str] = []

    # Collect prior art card if rendered
    comparison_card = state.get("comparison_card_path", "")
    if comparison_card and Path(comparison_card).exists():
        image_paths.append(comparison_card)

    try:
        # 1. Try PaperBanana (premium path)
        import paperbanana as pb  # type: ignore[import]
        from pathlib import Path as _Path

        output_dir = _Path("./output/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = str(output_dir / f"diagram_{run_id}.png")

        agent = pb.PaperBananaAgent(api_key=settings.google_api_key)
        diagram_path = agent.generate_architecture_diagram(
            title=paper.get("title", ""),
            methodology_text=analysis.get("methodology", ""),
            output_path=filename,
            style="cyberpunk_dark",
        )
        image_paths.insert(0, diagram_path)
        logger.info("paperbanana_success", path=diagram_path)

    except ImportError:
        # 2. Fallback: html2image + enhanced Jinja2 template
        logger.warning("paperbanana_sdk_missing", hint="Falling back to html2image cyberpunk card.")

        try:
            from html2image import Html2Image  # type: ignore[import]
            from jinja2 import Environment, FileSystemLoader, select_autoescape

            TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
            OUTPUT_DIR = Path("./output/images")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # ── Build gauge HTML (Feature 8) ──────────────────────
            gauges_html = ""
            if research_scores:
                gauge_defs = [
                    ("Novelty",       research_scores.get("novelty", 0),              "#00f3ff"),
                    ("Clarity",       research_scores.get("methodology_clarity", 0),  "#9d00ff"),
                    ("Benchmarks",    research_scores.get("benchmark_improvement", 0),"#00ff9d"),
                    ("Repro",         research_scores.get("reproducibility", 0),       "#ff2d78"),
                ]
                gauge_svgs = [_render_gauge_svg(lbl, val, col) for lbl, val, col in gauge_defs]
                gauges_html = "".join(gauge_svgs)

            # ── Base64-encode benchmark chart (Feature 5) ──────────
            benchmark_chart_b64 = ""
            if benchmark_chart_path and Path(benchmark_chart_path).exists():
                benchmark_chart_b64 = base64.b64encode(
                    Path(benchmark_chart_path).read_bytes()
                ).decode()

            # ── Render enhanced card ────────────────────────────────
            hti = Html2Image(
                output_path=str(OUTPUT_DIR),
                size=(1200, 800),
                custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
            )
            env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=select_autoescape(["html"]),
            )
            template = env.get_template("research_card.html")
            html = template.render(
                title=paper.get("title", "Deep Tech Research Update"),
                core_problem=analysis.get("core_problem", ""),
                methodology=analysis.get("methodology", ""),
                run_id=run_id[:8],
                gauges_html=gauges_html,
                benchmark_chart_b64=benchmark_chart_b64,
                architecture_diagram_b64=arch_b64,
                architecture_fallback_text=arch_fallback,
            )

            filename = f"research_card_{run_id}.png"
            hti.screenshot(html_str=html, save_as=filename)
            image_paths.insert(0, str(OUTPUT_DIR / filename))
            logger.info("fallback_image_generated", path=filename)

        except Exception as fallback_error:
            logger.error("fallback_image_gen_failed", error=str(fallback_error))

    except Exception as e:
        logger.error("visual_generation_failed", error=str(e))

    return {"image_paths": image_paths, "current_step": "visuals_generated"}


# ── 2. Publishing Nodes ──────────────────────────────────────────────────────

def _publish_research_node(state: PipelineState) -> dict:
    """Send email newsletter + publish LinkedIn carousel (document post)."""
    from app.services.email_service import EmailService

    run_id = state["run_id"]
    image_paths = state.get("image_paths", [])

    logger.info(
        "publishing_research",
        run_id=run_id,
        linkedin_chars=len(state.get("linkedin_draft", "")),
    )

    # ── Email newsletter ──────────────────────────────────────────────────
    # Include carousel slides in the newsletter alongside the cards
    research_slides = state.get("research_carousel_slide_paths") or []
    newsletter_attachments = image_paths + [
        p for p in research_slides if Path(p).exists()
    ] or None

    try:
        EmailService().send_newsletter(
            html_content=state.get("newsletter_html", ""),
            subject="🔬 AI Research Analyst: Deep Dive",
            image_paths=newsletter_attachments,
        )
        logger.info("research_newsletter_sent", run_id=run_id)
    except Exception as e:
        logger.error("research_newsletter_send_failed", run_id=run_id, error=str(e))

    # ── LinkedIn PDF carousel (Feature 2) ─────────────────────────────────
    carousel_pdf = state.get("research_carousel_pdf_path", "")
    linkedin_draft = state.get("linkedin_draft", "")
    paper = state.get("chosen_research_paper", {})

    if carousel_pdf and Path(carousel_pdf).exists() and linkedin_draft:
        try:
            from app.services.linkedin_service import LinkedInService

            # Strip hashtags from commentary (links in first comment strategy)
            commentary = linkedin_draft
            LinkedInService().publish_document_post(
                text=commentary,
                pdf_path=carousel_pdf,
                title=paper.get("title", "AI Research Deep Dive")[:100],
            )
            logger.info("research_carousel_published_to_linkedin", run_id=run_id)
        except Exception as e:
            logger.error("research_linkedin_publish_failed", run_id=run_id, error=str(e))

    return {"current_step": "published"}


def _revise_research_node(state: PipelineState) -> dict:
    return {"current_step": "revising"}


def _route_after_approval(state: PipelineState) -> Literal["publish", "revise"]:
    if state.get("approval_status") == "approved":
        return "publish"
    return "revise"


# ── 3. Build the Graph ───────────────────────────────────────────────────────

def build_research_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(PipelineState)

    # Scrape ArXiv
    workflow.add_node("scrape_arxiv", scrape_arxiv_node)

    # Intelligence pipeline
    workflow.add_node("select_paper",         select_paper_node)
    workflow.add_node("deep_analysis",        deep_analysis_node)
    workflow.add_node("score_research",       score_research_node)       # F8: gauges
    workflow.add_node("score_hook",           score_hook_node)            # F1: hook quality
    workflow.add_node("benchmark_chart",      benchmark_chart_node)       # F5: bar chart
    workflow.add_node("architecture_diagram", architecture_diagram_node)  # F6: PDF figure
    workflow.add_node("prior_art",            prior_art_node)             # F7: comparison card
    workflow.add_node("research_carousel",    research_carousel_node)     # F2: PDF carousel

    # Visuals + HITL + Publish
    workflow.add_node("paperbanana_visual", paperbanana_visual_node)
    workflow.add_node("human_approval",     human_approval_node)
    workflow.add_node("publish",            _publish_research_node)
    workflow.add_node("revise",             _revise_research_node)

    # Edges
    workflow.add_edge(START,                "scrape_arxiv")
    workflow.add_edge("scrape_arxiv",       "select_paper")
    workflow.add_edge("select_paper",       "deep_analysis")
    workflow.add_edge("deep_analysis",      "score_research")
    workflow.add_edge("score_research",     "score_hook")
    workflow.add_edge("score_hook",         "benchmark_chart")
    workflow.add_edge("benchmark_chart",    "architecture_diagram")
    workflow.add_edge("architecture_diagram", "prior_art")
    workflow.add_edge("prior_art",          "research_carousel")
    workflow.add_edge("research_carousel",  "paperbanana_visual")
    workflow.add_edge("paperbanana_visual", "human_approval")

    workflow.add_conditional_edges("human_approval", _route_after_approval)
    workflow.add_edge("publish", END)
    workflow.add_edge("revise", "deep_analysis")

    if checkpointer is None:
        checkpointer = InMemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("research_graph_compiled", node_count=len(workflow.nodes))
    return app

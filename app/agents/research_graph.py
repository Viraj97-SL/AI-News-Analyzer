from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.approval import human_approval_node
from app.agents.nodes.scraper import scrape_arxiv_node
from app.agents.state import PipelineState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Pydantic Schemas for Structured LLM Output ──
class PaperSelection(BaseModel):
    chosen_url: str = Field(description="The exact URL of the chosen paper.")
    reasoning: str = Field(description="1-sentence reason why this is the most impactful.")


class DeepAnalysis(BaseModel):
    core_problem: str = Field(description="What specific gap or problem is this paper solving?")
    methodology: str = Field(description="How did they solve it? Detail the architecture or math.")
    breakthroughs: str = Field(description="What were the quantifiable results or benchmarks?")
    limitations: str = Field(description="What are the drawbacks or future work needed?")


# ── 1. Intelligence Nodes ──
def select_paper_node(state: PipelineState) -> dict:
    """Uses Gemini Flash to filter ArXiv results and select the single best paper."""
    logger.info("research_node_running", step="selecting_best_paper")
    articles = state.get("raw_articles", [])

    if not articles:
        return {"current_step": "no_papers_found"}

    # Format abstracts for the LLM
    papers_text = "\n\n".join([
        f"URL: {a['url']}\nTitle: {a['title']}\nAbstract: {a['content'][:1000]}"
        for a in articles[:30]  # Limit to top 30 to save context window
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=settings.google_api_key
    ).with_structured_output(PaperSelection)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Principal Investigator of an elite AI lab. Read these recent ArXiv paper abstracts. Select the SINGLE most groundbreaking, highly-novel paper that engineers and researchers absolutely must know about. Favor novel architectures and major benchmark breakthroughs over minor optimizations."),
        ("user", "{papers}")
    ])

    chain = prompt | llm
    result = chain.invoke({"papers": papers_text})

    # Find the full article object that matches the chosen URL
    chosen_paper = next((a for a in articles if a["url"] == result.chosen_url), articles[0])

    logger.info("paper_selected", title=chosen_paper["title"])
    return {
        "chosen_research_paper": chosen_paper,
        "current_step": "paper_selected"
    }


def deep_analysis_node(state: PipelineState) -> dict:
    """Uses Gemini Pro to extract thematic analysis and methodology."""
    logger.info("research_node_running", step="deep_analysis")
    paper = state.get("chosen_research_paper")

    if not paper:
        return {"current_step": "error_no_paper"}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.4,
        api_key=settings.google_api_key
    ).with_structured_output(DeepAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Senior AI Research Scientist. Analyze the following paper. Extract the deep technical concepts, explaining them clearly for a highly technical audience (PhDs, ML Engineers). Do not use fluff or generic marketing speak."),
        ("user", "Title: {title}\n\nContent: {content}")
    ])

    analysis = (prompt | llm).invoke({
        "title": paper["title"],
        "content": paper["content"]
    })

    # 1. Draft the LinkedIn Post
    linkedin_draft = f"""🚨 Deep Tech Breakdown: {paper['title']}

The Core Problem:
{analysis.core_problem}

The Methodology:
{analysis.methodology}

The Breakthrough:
{analysis.breakthroughs}

Limitations to consider: {analysis.limitations}

Read the full paper here: {paper['url']}
#AIResearch #MachineLearning #DeepLearning #ArXiv
"""

    # 2. Draft the Newsletter HTML
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

    # We convert the Pydantic object to a dict to store in LangGraph state
    return {
        "deep_analysis": analysis.model_dump(),
        "linkedin_draft": linkedin_draft,
        "newsletter_html": newsletter_html,
        "current_step": "analysis_complete"
    }


def paperbanana_visual_node(state: PipelineState) -> dict:
    """Uses PaperBanana (if available) OR falls back to html2image for visuals."""
    logger.info("research_node_running", step="generating_visual")
    analysis = state.get("deep_analysis", {})
    paper = state.get("chosen_research_paper", {})
    run_id = state.get("run_id", "test")

    image_paths = []

    try:
        # 1. Try the cutting-edge PaperBanana integration first
        import paperbanana as pb
        from pathlib import Path

        output_dir = Path("./output/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = str(output_dir / f"diagram_{run_id}.png")

        agent = pb.PaperBananaAgent(api_key=settings.google_api_key)
        diagram_path = agent.generate_architecture_diagram(
            title=paper.get("title", ""),
            methodology_text=analysis.get("methodology", ""),
            output_path=filename,
            style="cyberpunk_dark"
        )

        image_paths.append(diagram_path)
        logger.info("paperbanana_success", path=diagram_path)

    except ImportError:
        # 2. THE FALLBACK: If PaperBanana isn't installed, use our existing html2image engine!
        logger.warning("paperbanana_sdk_missing", hint="Falling back to html2image Cyberpunk card.")

        try:
            from html2image import Html2Image
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            from pathlib import Path

            # Using 2 parents to properly target /app/templates
            TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
            OUTPUT_DIR = Path("./output/images")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            hti = Html2Image(
                output_path=str(OUTPUT_DIR),
                size=(1200, 627),
                custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
            )

            env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=select_autoescape(["html"]),
            )

            # 1. Point to the NEW template
            template = env.get_template("research_card.html")

            # 2. Pass the deep analysis fields directly into the template
            html = template.render(
                title=paper.get("title", "Deep Tech Research Update"),
                core_problem=analysis.get("core_problem", "No problem specified."),
                methodology=analysis.get("methodology", "No methodology specified."),
                run_id=run_id[:8]  # Just use first 8 chars of run_id to keep it clean
            )

            filename = f"research_card_{run_id}.png"
            hti.screenshot(html_str=html, save_as=filename)
            image_paths.append(str(OUTPUT_DIR / filename))
            logger.info("fallback_image_generated", path=filename)

        except Exception as fallback_error:
            logger.error("fallback_image_gen_failed", error=str(fallback_error))

    except Exception as e:
        logger.error("visual_generation_failed", error=str(e))

    return {"image_paths": image_paths, "current_step": "visuals_generated"}


# ── 2. Publishing Nodes ──
def _publish_research_node(state: PipelineState) -> dict:
    """Send email newsletter and publish LinkedIn post."""
    from app.services.email_service import EmailService

    run_id = state["run_id"]
    image_paths = state.get("image_paths", [])

    logger.info(
        "publishing_research",
        run_id=run_id,
        linkedin_chars=len(state.get("linkedin_draft", "")),
    )

    try:
        EmailService().send_newsletter(
            html_content=state.get("newsletter_html", ""),
            subject="🔬 AI Research Analyst: Deep Dive",
            image_paths=image_paths or None,
        )
        logger.info("research_newsletter_sent", run_id=run_id)
    except Exception as e:
        logger.error("research_newsletter_send_failed", run_id=run_id, error=str(e))

    return {"current_step": "published"}


def _revise_research_node(state: PipelineState) -> dict:
    return {"current_step": "revising"}


def _route_after_approval(state: PipelineState) -> Literal["publish", "revise"]:
    if state.get("approval_status") == "approved":
        return "publish"
    return "revise"


# ── 3. Build the Graph ──
def build_research_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(PipelineState)

    # Scrape strictly ArXiv
    workflow.add_node("scrape_arxiv", scrape_arxiv_node)

    # Intelligence Pipeline
    workflow.add_node("select_paper", select_paper_node)
    workflow.add_node("deep_analysis", deep_analysis_node)
    workflow.add_node("paperbanana_visual", paperbanana_visual_node)

    # HITL & Publish
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("publish", _publish_research_node)
    workflow.add_node("revise", _revise_research_node)

    # Edges
    workflow.add_edge(START, "scrape_arxiv")
    workflow.add_edge("scrape_arxiv", "select_paper")
    workflow.add_edge("select_paper", "deep_analysis")
    workflow.add_edge("deep_analysis", "paperbanana_visual")
    workflow.add_edge("paperbanana_visual", "human_approval")

    workflow.add_conditional_edges("human_approval", _route_after_approval)
    workflow.add_edge("publish", END)
    workflow.add_edge("revise", "deep_analysis")  # Re-analyze on rejection

    if checkpointer is None:
        checkpointer = InMemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("research_graph_compiled", node_count=len(workflow.nodes))
    return app
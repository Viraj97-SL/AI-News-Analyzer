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
from app.agents.nodes.manual_papers import load_manual_papers_node
from app.agents.nodes.paper_ranker import rank_papers_node
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


class RichDeepAnalysis(BaseModel):
    # ── Core fields (kept for backward compat — consumed by benchmark_chart, prior_art nodes) ──
    core_problem: str = Field(description=(
        "3-5 sentences: what specific gap or unsolved problem is this paper addressing? "
        "Be concrete about what previous approaches failed at."
    ))
    methodology: str = Field(description=(
        "4-6 sentences: the core architecture, algorithm, or training procedure. "
        "Include specific design choices and why they differ from prior work."
    ))
    breakthroughs: str = Field(description=(
        "3-5 sentences: quantifiable results with specific numbers. "
        "Name the benchmarks and state the delta over prior SOTA."
    ))
    limitations: str = Field(description=(
        "3-4 sentences: honest drawbacks, assumptions that may not hold, "
        "failure modes, and what future work is explicitly needed."
    ))

    # ── Enriched fields for deeper analysis ──
    executive_summary: str = Field(description=(
        "Two paragraphs of plain-English summary accessible to any software engineer. "
        "Paragraph 1: what problem and why it matters. "
        "Paragraph 2: how they solved it and what changed as a result."
    ))
    key_contributions: list[str] = Field(
        description=(
            "3-5 specific contributions, each a single sentence with concrete details. "
            "Example: 'Sparse attention mask that reduces memory from O(n²) to O(n log n), "
            "enabling 4× longer context windows at the same compute budget.'"
        ),
        default_factory=list,
    )
    technical_innovation: str = Field(description=(
        "3-4 sentences on what is genuinely NEW vs prior work. "
        "Specify what changed architecturally or algorithmically vs the closest competing method."
    ))
    experiment_setup: str = Field(description=(
        "3-5 sentences: which datasets were used, which baselines were compared against, "
        "compute budget, training details, and evaluation protocol. "
        "If only the abstract is available, note what details are missing."
    ))
    quantitative_results: list[str] = Field(
        description=(
            "4-6 key metrics as formatted strings: "
            "'<MetricName>: <value> (<delta> vs <prior>, <context>)'. "
            "Example: 'MMLU: 89.2% (+3.1% vs GPT-4-Turbo, 5-shot prompting)'. "
            "Return empty list if no specific numbers are present."
        ),
        default_factory=list,
    )
    ablation_highlights: str = Field(description=(
        "3-4 sentences: what components were ablated, what happened when each was removed, "
        "and what this proves about the design choices. "
        "If the paper has no ablation study, state that clearly."
    ))
    real_world_applications: list[str] = Field(
        description=(
            "3 concrete, specific use cases this research enables or improves. "
            "Not generic 'AI applications' — name the actual product or workflow."
        ),
        default_factory=list,
    )
    ecosystem_impact: str = Field(description=(
        "2-3 sentences: which frameworks, libraries, or products are directly affected. "
        "What does a PyTorch or HuggingFace practitioner need to know?"
    ))
    expert_interpretation: str = Field(description=(
        "3-4 sentences: what does this mean for ML practitioners TODAY? "
        "What should an engineer do differently after reading this? "
        "What should they watch for in follow-up work?"
    ))
    technical_deep_dive: str = Field(description=(
        "400-500 words of detailed explanation for a PhD-level reader. "
        "Cover: specific architecture choices, key equations in plain text, "
        "training tricks, and WHY each design decision was made. "
        "Do not include citations or URLs."
    ))
    future_directions: list[str] = Field(
        description=(
            "3-5 specific open research questions or natural follow-on directions "
            "that this paper creates or leaves unanswered."
        ),
        default_factory=list,
    )
    significance_verdict: str = Field(description=(
        "Exactly one of: 'Incremental', 'Solid Contribution', 'Major Contribution', 'Paradigm Shift'. "
        "Calibration: Transformers paper = Paradigm Shift, BERT = Major Contribution, "
        "LoRA = Solid Contribution, minor hyperparameter tuning = Incremental."
    ))


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
    """
    Select the single best paper to deep-dive into.

    Primary path: use pre-computed paper_rankings (from rank_papers_node).
      Manual papers always rank first due to the composite-score boost applied
      in rank_papers_node, so they are always selected when present.

    Fallback: original LLM-picks-from-abstracts approach when rankings are empty.
    """
    logger.info("research_node_running", step="selecting_best_paper")
    articles = state.get("raw_articles", [])
    rankings = state.get("paper_rankings", [])

    if not articles:
        return {"current_step": "no_papers_found"}

    url_to_article = {a["url"]: a for a in articles}

    # ── Primary: use ranking ───────────────────────────────────────────
    if rankings:
        for ranked in rankings:
            article = url_to_article.get(ranked.get("paper_url", ""))
            if article:
                logger.info(
                    "paper_selected_by_ranking",
                    title=article["title"],
                    score=ranked.get("composite_score", 0),
                    source=article.get("source", "arxiv"),
                    is_manual=ranked.get("is_manual", False),
                )
                return {"chosen_research_paper": article, "current_step": "paper_selected"}

    # ── Fallback: LLM selection ────────────────────────────────────────
    logger.info("using_llm_fallback_selection", reason="no_rankings_available")
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
    logger.info("paper_selected_by_llm", title=chosen_paper["title"])
    return {"chosen_research_paper": chosen_paper, "current_step": "paper_selected"}


_LINKEDIN_RESEARCH_SYSTEM = """\
You are a LinkedIn content strategist for an elite AI research lab with 150k+ followers.

Write a LinkedIn post about this research paper. Use this EXACT 8-section structure:

─── HOOK (≤210 chars — first 2-3 lines) ───
Open with a SPECIFIC metric or bold claim that stops the scroll.
BAD: "This new AI paper is fascinating!"
GOOD: "New paper beats GPT-4 on reasoning by 23% — using 10× less compute."

[blank line]

─── CONTEXT (2 sentences) ───
Why was this problem unsolved? What did prior approaches get wrong?

[blank line]

─── CORE IDEA (3-4 sentences) ───
Plain-English methodology. Specific enough for an ML engineer to grasp the key insight. No jargon without explanation.

[blank line]

─── KEY RESULTS (4-5 lines) ───
Format: → <Metric>: <value> (<comparison if available>)
Example: → MMLU: 89.2% (+3.1% vs GPT-4-Turbo)

[blank line]

─── PRACTITIONER IMPACT (2 sentences) ───
What changes for engineers TODAY? Be concrete — name a framework or workflow.

[blank line]

─── HONEST CAVEAT (1-2 sentences) ───
One real limitation. Don't cheerleader. Credibility comes from honesty.

[blank line]

─── QUESTION (1 line) ───
Specific CTA tied to THIS paper. Not "What do you think?" but a real tradeoff question.
Example: "Would you trade 5% accuracy for 10× inference speed in production?"

[blank line]

─── HASHTAGS ───
5 max: #AIResearch #MachineLearning + 3 topic-specific

[blank line]

🔗 Full paper in the first comment 👇

HARD RULES:
- Total: 1,800–2,400 characters
- NO URLs in body
- NO filler: "game-changer", "revolutionary", "groundbreaking", "fascinating"
- Every claim must come from the actual paper results
- Tone: senior researcher explaining to senior engineer — authoritative, not hype\
"""


def _build_research_article_html(paper: dict, analysis: "RichDeepAnalysis") -> str:  # type: ignore[name-defined]
    """Generate a structured full-length HTML research article for the email newsletter."""
    title = paper.get("title", "Research Deep Dive")
    url = paper.get("url", "")

    verdict_colors = {
        "Paradigm Shift": ("#00ff9d", "#003322"),
        "Major Contribution": ("#00f3ff", "#001a22"),
        "Solid Contribution": ("#9d00ff", "#1a0022"),
        "Incremental": ("#ff2d78", "#220011"),
    }
    verdict = analysis.significance_verdict
    v_color, v_bg = verdict_colors.get(verdict, ("#aaa", "#111"))

    contributions_html = "".join(
        f'<li style="margin-bottom:10px">{c}</li>' for c in analysis.key_contributions
    )
    results_html = "".join(
        f'<li style="margin-bottom:8px;font-family:monospace;font-size:14px">{r}</li>'
        for r in analysis.quantitative_results
    )
    applications_html = "".join(
        f'<li style="margin-bottom:10px">{a}</li>' for a in analysis.real_world_applications
    )
    directions_html = "".join(
        f'<li style="margin-bottom:8px">{d}</li>' for d in analysis.future_directions
    )

    return f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:700px;
            margin:0 auto;color:#1a1a2e;line-height:1.75;background:#fff">

  <!-- Header banner -->
  <div style="background:linear-gradient(135deg,#0a0a1a,#1a0a2e);color:white;
              padding:36px 40px;border-radius:12px;margin-bottom:32px">
    <div style="font-family:monospace;font-size:10px;color:#00f3ff;letter-spacing:2px;
                text-transform:uppercase;margin-bottom:12px">
      AI Research Analyst · Research Deep Dive
    </div>
    <h1 style="font-size:22px;font-weight:800;margin:0 0 16px;line-height:1.3;color:#fff">
      {title}
    </h1>
    <div style="display:inline-block;background:{v_bg};border:1px solid {v_color};
                color:{v_color};font-family:monospace;font-size:11px;padding:4px 12px;
                border-radius:4px;letter-spacing:1px">
      VERDICT: {verdict.upper()}
    </div>
    <div style="margin-top:16px;font-size:12px;color:rgba(255,255,255,0.45)">
      🔗 <a href="{url}" style="color:#00f3ff">{url}</a>
    </div>
  </div>

  <!-- Executive Summary -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #00f3ff;
             padding-left:14px;margin:0 0 12px">Executive Summary</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.executive_summary}</p>

  <!-- The Problem -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #9d00ff;
             padding-left:14px;margin:0 0 12px">The Problem</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.core_problem}</p>

  <!-- What's Genuinely New -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #ff2d78;
             padding-left:14px;margin:0 0 12px">What's Genuinely New</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.technical_innovation}</p>

  <!-- Key Contributions -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #00ff9d;
             padding-left:14px;margin:0 0 12px">Key Contributions</h2>
  <ol style="margin:0 0 28px;padding-left:22px;color:#333;font-size:15px">
    {contributions_html}
  </ol>

  <!-- Technical Deep Dive -->
  <div style="background:#f8f9ff;border:1px solid #e0e8ff;border-radius:10px;
              padding:28px 32px;margin-bottom:28px">
    <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;margin:0 0 14px">
      Technical Deep Dive
    </h2>
    <p style="margin:0;color:#333;font-size:15px;white-space:pre-line">{analysis.technical_deep_dive}</p>
  </div>

  <!-- Experiment Setup -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #00f3ff;
             padding-left:14px;margin:0 0 12px">Experiment Setup</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.experiment_setup}</p>

  <!-- Quantitative Results -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #00ff9d;
             padding-left:14px;margin:0 0 12px">Results at a Glance</h2>
  <ul style="margin:0 0 28px;padding-left:20px;list-style:none;color:#333">
    {results_html if results_html else '<li style="color:#999;font-style:italic">See full paper for detailed results.</li>'}
  </ul>

  <!-- Ablation Insights -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #9d00ff;
             padding-left:14px;margin:0 0 12px">Ablation Study Insights</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.ablation_highlights}</p>

  <!-- Real-World Applications -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #ff2d78;
             padding-left:14px;margin:0 0 12px">Real-World Applications</h2>
  <ul style="margin:0 0 28px;padding-left:22px;color:#333;font-size:15px">
    {applications_html}
  </ul>

  <!-- Ecosystem Impact -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #00f3ff;
             padding-left:14px;margin:0 0 12px">Ecosystem Impact</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.ecosystem_impact}</p>

  <!-- Expert Interpretation -->
  <div style="background:linear-gradient(135deg,rgba(0,243,255,0.05),rgba(157,0,255,0.05));
              border:1px solid rgba(0,243,255,0.2);border-radius:10px;
              padding:28px 32px;margin-bottom:28px">
    <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;margin:0 0 14px">
      What This Means for You
    </h2>
    <p style="margin:0;color:#333;font-size:15px">{analysis.expert_interpretation}</p>
  </div>

  <!-- Limitations -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #ff2d78;
             padding-left:14px;margin:0 0 12px">Honest Limitations</h2>
  <p style="margin:0 0 28px;color:#333;font-size:15px">{analysis.limitations}</p>

  <!-- Future Directions -->
  <h2 style="font-size:18px;font-weight:700;color:#0a0a1a;border-left:4px solid #9d00ff;
             padding-left:14px;margin:0 0 12px">What Comes Next</h2>
  <ul style="margin:0 0 36px;padding-left:22px;color:#333;font-size:15px">
    {directions_html}
  </ul>

  <!-- Footer -->
  <div style="background:#f5f5f5;border-radius:8px;padding:20px 24px;
              font-size:13px;color:#666;text-align:center">
    AI Research Analyst · Powered by Gemini 2.5 Pro ·
    <a href="{url}" style="color:#0a66c2">Read the full paper →</a>
  </div>
</div>
"""


def deep_analysis_node(state: PipelineState) -> dict:
    """Deep analysis with Gemini Pro: extracts 16-field rich analysis, LLM LinkedIn draft, full article HTML."""
    logger.info("research_node_running", step="deep_analysis")
    paper = state.get("chosen_research_paper")

    if not paper:
        return {"current_step": "error_no_paper"}

    # ── 1. Rich structured analysis (Gemini 2.5 Pro) ──────────────────────
    analysis_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.3,
        api_key=settings.google_api_key,
    ).with_structured_output(RichDeepAnalysis)

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Senior AI Research Scientist and Principal Investigator. "
         "Analyse this research paper deeply. Extract technical concepts with precision, "
         "explaining them for a highly technical audience (PhDs, ML Engineers, Staff Engineers). "
         "Do NOT use marketing speak or generic praise. Be specific, calibrated, and honest. "
         "If the content is only an abstract, extract what you can and clearly note where "
         "full-paper details are needed."),
        ("user", "Title: {title}\n\nPaper Content:\n{content}"),
    ])

    try:
        analysis: RichDeepAnalysis = (analysis_prompt | analysis_llm).invoke({
            "title": paper["title"],
            "content": paper["content"],
        })
    except Exception as e:
        logger.error("rich_analysis_failed", error=str(e))
        return {"current_step": "error_analysis_failed"}

    # ── 2. LinkedIn draft via LLM (Gemini 2.5 Flash) ─────────────────────
    flash_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=settings.google_api_key,
    )

    linkedin_prompt = ChatPromptTemplate.from_messages([
        ("system", _LINKEDIN_RESEARCH_SYSTEM),
        ("user",
         "Paper title: {title}\n\n"
         "Core problem: {core_problem}\n\n"
         "Methodology: {methodology}\n\n"
         "Key contributions:\n{contributions}\n\n"
         "Quantitative results:\n{results}\n\n"
         "Practitioner impact: {expert_interpretation}\n\n"
         "Limitations: {limitations}\n\n"
         "Significance verdict: {verdict}"),
    ])

    contributions_text = "\n".join(f"• {c}" for c in analysis.key_contributions)
    results_text = "\n".join(f"• {r}" for r in analysis.quantitative_results)

    try:
        linkedin_response = (linkedin_prompt | flash_llm).invoke({
            "title": paper["title"],
            "core_problem": analysis.core_problem,
            "methodology": analysis.methodology,
            "contributions": contributions_text or "See methodology above.",
            "results": results_text or analysis.breakthroughs,
            "expert_interpretation": analysis.expert_interpretation,
            "limitations": analysis.limitations,
            "verdict": analysis.significance_verdict,
        })
        raw = linkedin_response.content
        linkedin_draft = (
            "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw).strip()
            if isinstance(raw, list) else raw.strip()
        )
    except Exception as e:
        logger.warning("linkedin_draft_llm_failed", error=str(e), fallback="using_template")
        linkedin_draft = (
            f"New research: {paper['title'][:80]}\n\n"
            f"{analysis.core_problem[:200]}\n\n"
            f"Key finding: {analysis.breakthroughs[:200]}\n\n"
            f"What does this mean for your work?\n\n"
            f"🔗 Full paper in the first comment 👇\n\n"
            f"#AIResearch #MachineLearning #DeepLearning"
        )

    # ── 3. Full research article HTML for email newsletter ────────────────
    newsletter_html = _build_research_article_html(paper, analysis)

    logger.info(
        "deep_analysis_complete",
        verdict=analysis.significance_verdict,
        contributions=len(analysis.key_contributions),
        results=len(analysis.quantitative_results),
        linkedin_chars=len(linkedin_draft),
    )

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
            "methodology": analysis.get("methodology", "") + "\n\n" + analysis.get("technical_innovation", ""),
            "breakthroughs": analysis.get("breakthroughs", "") + "\n\n" + "\n".join(analysis.get("quantitative_results", [])),
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
            _hook_resp = (rewrite_prompt | flash).invoke({
                "weakness": score.reasoning,
                "hook": hook,
                "full_post": linkedin_draft[:600],
            }).content
            new_hook = (
                "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in _hook_resp).strip()
                if isinstance(_hook_resp, list) else _hook_resp.strip()
            )

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
    paper_figures = state.get("paper_figures", [])
    arch_caption = paper_figures[0].get("caption", "") if paper_figures else ""
    extra_figures = paper_figures[1:3] if len(paper_figures) > 1 else []

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
                arch_caption=arch_caption,
                extra_figures=extra_figures,
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

    # ── Paper sourcing (fan-out; both feed raw_articles via Annotated reducer) ──
    workflow.add_node("scrape_arxiv",       scrape_arxiv_node)
    workflow.add_node("load_manual_papers", load_manual_papers_node)

    # ── Ranking + selection ───────────────────────────────────────────────────
    workflow.add_node("rank_papers",        rank_papers_node)
    workflow.add_node("select_paper",       select_paper_node)

    # ── Intelligence pipeline ─────────────────────────────────────────────────
    workflow.add_node("deep_analysis",        deep_analysis_node)
    workflow.add_node("score_research",       score_research_node)       # F8: gauges
    workflow.add_node("score_hook",           score_hook_node)            # F1: hook quality
    workflow.add_node("benchmark_chart",      benchmark_chart_node)       # F5: bar chart
    workflow.add_node("architecture_diagram", architecture_diagram_node)  # F6: figures
    workflow.add_node("prior_art",            prior_art_node)             # F7: comparison card
    workflow.add_node("research_carousel",    research_carousel_node)     # F2: PDF carousel

    # ── Visuals + HITL + Publish ──────────────────────────────────────────────
    workflow.add_node("paperbanana_visual", paperbanana_visual_node)
    workflow.add_node("human_approval",     human_approval_node)
    workflow.add_node("publish",            _publish_research_node)
    workflow.add_node("revise",             _revise_research_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Fan-out from START so scrapers run in parallel
    workflow.add_edge(START, "scrape_arxiv")
    workflow.add_edge(START, "load_manual_papers")
    # Both scrapers must finish before ranking (LangGraph waits automatically)
    workflow.add_edge("scrape_arxiv",       "rank_papers")
    workflow.add_edge("load_manual_papers", "rank_papers")
    workflow.add_edge("rank_papers",        "select_paper")
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

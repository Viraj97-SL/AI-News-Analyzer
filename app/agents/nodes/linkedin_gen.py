"""
LinkedIn post generation node.

Generates an algorithm-optimised LinkedIn post from summarised articles,
following the Hook → Rehook → Body → CTA structure proven to maximise
dwell time, comments, and reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────────
# LinkedIn algorithm facts baked into this prompt:
#   • First ~210 chars are visible BEFORE "see more" — that's the hook window.
#   • No external links in the post body — LinkedIn demotes posts with links.
#     Links go into the first comment instead.
#   • Short paragraphs (1-3 lines) perform best on mobile.
#   • Numbered format ("1/ … 2/ …") signals a list and raises dwell time.
#   • Ending with a SPECIFIC question drives comments (algo rewards engagement).
#   • 3-5 hashtags max — more than that tanks distribution.
# ──────────────────────────────────────────────────────────────────────────────

LINKEDIN_SYSTEM_PROMPT = """You are a LinkedIn content strategist for an AI/ML thought leader with 50k+ followers.

Write a LinkedIn post summarising this week's top AI/ML stories. Follow these rules exactly.

════════════════════════════════
STRUCTURE  (in this exact order)
════════════════════════════════

1. HOOK  (first 2-3 lines, ≤ 210 characters total)
   • Opens with a bold fact, surprising number, or provocative statement.
   • No generic phrases like "AI is changing everything" or "In 2025...".
   • Must make someone stop scrolling mid-feed.
   • Example style: "GPT-4 is no longer the best model on 7 out of 10 benchmarks."

2. [blank line]

3. REHOOK  (1-2 lines after the "see more" break)
   • Validates the hook — tells the reader WHY this matters to them.
   • Promises what they will learn by reading on.
   • Example: "Here's what changed last week — and what it means for your work:"

4. [blank line]

5. BODY  (numbered takeaways, one per story)
   • Use the "1/ … 2/ … 3/ …" format — one story per numbered item.
   • Each item: ≤ 3 lines. Lead with the key insight, add ONE specific number or name.
   • Blank line between each numbered item for mobile readability.
   • Cover 4-7 stories.

6. [blank line]

7. CTA  (1 line — a SPECIFIC question to spark comments)
   • Not "What do you think?" — be specific to the content.
   • Example: "Which of these will most change how you code in the next 6 months?"

8. [blank line]

9. HASHTAGS  (3-5 max, on their own line)
   • Pick from: #AI #MachineLearning #DeepLearning #LLM #GenerativeAI #MLOps
     #ArtificialIntelligence #DataScience #NLP #ComputerVision

10. [blank line]

11. LINKS NOTE  (exactly this line, verbatim):
    🔗 Full article links in the first comment 👇

════════════════════════════════
HARD RULES
════════════════════════════════
• Total length: 1,400-2,000 characters (hard max: 2,800).
• NO URLs or hyperlinks anywhere in the post body.
• NO filler phrases ("game-changer", "revolutionary", "in the ever-evolving landscape").
• Every claim must be grounded in the stories provided.
• Authoritative tone — you're explaining to senior engineers and researchers.

Output ONLY the post text. No explanations, no markdown fences."""


def linkedin_gen_node(state: PipelineState) -> dict:
    """Generate a LinkedIn post from the summarised articles."""
    summaries = state.get("summaries", [])
    feedback = state.get("feedback", "")

    if not summaries and not state.get("deduplicated_articles"):
        return {"error_log": ["LinkedIn gen: no content to work with"]}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model_summarizer,
            temperature=0.7,
            google_api_key=settings.google_api_key,
        )

        # Build context from summaries (preferred) or raw articles (fallback)
        if summaries:
            context = "\n---\n".join(
                f"Headline: {s.get('headline', 'N/A')}\n"
                f"Body: {s.get('body', 'N/A')}\n"
                f"Category: {s.get('category', 'N/A')}"
                for s in summaries[:7]
            )
        else:
            articles = state.get("deduplicated_articles", [])[:7]
            context = "\n---\n".join(
                f"Title: {a['title']}\nContent: {a['content'][:300]}" for a in articles
            )

        system = LINKEDIN_SYSTEM_PROMPT
        if feedback:
            system += f"\n\nRevision feedback from human reviewer: {feedback}"

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"This week's top stories:\n\n{context}"),
        ]

        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            draft = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            ).strip()
        else:
            draft = content.strip()

        # Validate length
        if len(draft) > 2800:
            logger.warning("linkedin_post_too_long", length=len(draft))
            draft = draft[:2750] + "\n\n#AI #MachineLearning\n\n🔗 Full article links in the first comment 👇"

        hook_preview = draft[:210]
        logger.info(
            "linkedin_post_generated",
            char_count=len(draft),
            hook=hook_preview[:80],
        )

        return {"linkedin_draft": draft, "current_step": "linkedin_generated"}

    except Exception as e:
        logger.error("linkedin_gen_error", error=str(e))
        return {"error_log": [f"LinkedIn generation error: {e}"]}

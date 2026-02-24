"""
LinkedIn post generation node.

Generates an engaging LinkedIn post from summarised articles,
respecting the 3,000-char limit and 210-char "see more" hook.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.state import PipelineState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

LINKEDIN_SYSTEM_PROMPT = """You are a LinkedIn content strategist for an AI/ML thought leader.

Write a LinkedIn post summarising this week's top AI/ML news. Follow these rules strictly:

FORMAT:
- Total length: 1,200–1,800 characters (hard max: 3,000)
- First 210 characters = the HOOK (this is all people see before "see more")
  → Make it provocative, specific, and curiosity-inducing. No generic openings.
- Use liberal whitespace between sections for mobile readability
- Use → bullets for 3–5 key takeaways
- End with an engaging question to drive comments
- Add 3–5 relevant hashtags at the bottom

TONE:
- Authoritative but accessible — you're explaining to smart professionals, not academics
- Lead with the "so what" — why should someone in tech care?
- Include specific numbers, names, or dates where possible
- No filler phrases like "In the ever-evolving landscape of AI..."

Output ONLY the post text, no explanations or markdown fences."""


def linkedin_gen_node(state: PipelineState) -> dict:
    """Generate a LinkedIn post from the summarised articles."""
    summaries = state.get("summaries", [])
    feedback = state.get("feedback", "")

    if not summaries and not state.get("deduplicated_articles"):
        return {"error_log": ["LinkedIn gen: no content to work with"]}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model_summarizer,
            temperature=0.7,  # slightly more creative for social content
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
            system += f"\n\nRevision feedback: {feedback}"

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"This week's top stories:\n\n{context}"),
        ]

        response = llm.invoke(messages)
        draft = response.content.strip()

        # Validate length constraints
        if len(draft) > 3000:
            logger.warning("linkedin_post_too_long", length=len(draft))
            draft = draft[:2950] + "\n\n#AI #MachineLearning"

        logger.info(
            "linkedin_post_generated",
            char_count=len(draft),
            hook_length=len(draft[:210].split("\n")[0]),
        )

        return {"linkedin_draft": draft, "current_step": "linkedin_generated"}

    except Exception as e:
        logger.error("linkedin_gen_error", error=str(e))
        return {"error_log": [f"LinkedIn generation error: {e}"]}

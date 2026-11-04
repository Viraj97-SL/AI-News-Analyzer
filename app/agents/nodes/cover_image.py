"""
Optional AI-generated cover illustration for the research carousel (D8/D9).

Only the cover gets a generated image. Technical diagrams stay deterministic
(real paper figures → generated SVG spec), because image models invent
structure. The call uses the existing Google API key and a Gemini image model
(about $0.04 per image). On by default (D21); set
RESEARCH_COVER_IMAGE_ENABLED=false to turn it off.

Failures never block the run: the cover simply renders without artwork.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.trends import trend_labels
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("./output/images")
EST_COST_PER_IMAGE_USD = 0.04
_CONCEPT_MAX_CHARS = 300
_ERROR_MAX_CHARS = 300


def _settings() -> Any:
    return get_settings()


def build_cover_prompt(paper: dict[str, Any], analysis: dict[str, Any]) -> str:
    concept = (analysis.get("technical_innovation") or analysis.get("core_problem") or "")[
        :_CONCEPT_MAX_CHARS
    ]
    themes = ", ".join(trend_labels(paper.get("trend_tags") or [])) or "machine learning research"
    return (
        "Editorial illustration for the cover of a research-paper explainer.\n"
        f"Paper: {paper.get('title', '')}\n"
        f"Core idea: {concept}\n"
        f"Themes: {themes}\n\n"
        "Style: clean, modern, abstract isometric shapes and flowing connections that visually "
        "suggest the core idea; soft light background (#F1F5F9) with sky-blue (#0EA5E9) and "
        "violet (#7C3AED) accents; generous negative space; landscape 16:9 composition.\n"
        "Strict rules: no text, no letters, no numbers, no logos, no brand marks, "
        "no people's faces, no fake charts or equations."
    )


def extract_image_bytes(response: Any) -> bytes | None:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                data: bytes = inline.data
                return data
    return None


def _default_client() -> Any:
    from google import genai

    return genai.Client(api_key=get_settings().google_api_key)


def cover_image_node(state: PipelineState, client: Any | None = None) -> dict[str, Any]:
    settings = _settings()
    if not settings.research_cover_image_enabled:
        return {}
    paper = state.get("chosen_research_paper") or {}
    if not paper:
        return {}

    run_id = state.get("run_id", "dev")
    try:
        from google.genai import types

        client = client or _default_client()
        response = client.models.generate_content(
            model=settings.research_cover_image_model,
            contents=build_cover_prompt(paper, state.get("deep_analysis") or {}),
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
            ),
        )
        image = extract_image_bytes(response)
        if not image:
            raise ValueError("model returned no image")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"cover_art_{run_id}.png"
        path.write_bytes(image)
        logger.info("cover_image_generated", path=str(path), est_cost_usd=EST_COST_PER_IMAGE_USD)
        return {"cover_image_b64": base64.b64encode(image).decode()}
    except Exception as e:
        # Quota errors carry multi-KB JSON; keep the log/state entry readable.
        message = str(e)[:_ERROR_MAX_CHARS]
        logger.warning("cover_image_failed", error=message)
        return {"error_log": [f"Cover image error: {message}"]}

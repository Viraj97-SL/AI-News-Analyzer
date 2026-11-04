"""
Research-pipeline model selection with quota-aware fallback (decision D22).

Google retires model generations quickly (Gemini 2.5 now returns 404 for new
keys) and the best models (e.g. ``gemini-4-argon``) return 429 on free-tier keys.
Instead of hardcoding one name per node, each tier has an ordered preference
list in settings. ``resolve_research_models`` probes the list once per run
(tiny prompt, short timeout) and caches the first model that answers;
``research_model(tier)`` is what the nodes call.

Enabling billing therefore upgrades the pipeline to the top model automatically,
with no code change.
"""

from __future__ import annotations

from typing import Any, Literal

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

Tier = Literal["deep", "fast"]
_PROBE_TIMEOUT_MS = 30_000
_PROBE_PROMPT = "Reply with the single word OK."

_resolved: dict[str, str] = {}


def candidates(tier: Tier) -> list[str]:
    settings = get_settings()
    models = settings.research_deep_models if tier == "deep" else settings.research_fast_models
    return [m for m in models if m]


def research_model(tier: Tier) -> str:
    """Model name for a tier: the probed winner if resolved, else the first preference."""
    if tier in _resolved:
        return _resolved[tier]
    prefs = candidates(tier)
    if not prefs:
        raise ValueError(f"No research models configured for tier {tier!r}")
    return prefs[0]


def _default_client() -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=get_settings().google_api_key,
        http_options=types.HttpOptions(timeout=_PROBE_TIMEOUT_MS),
    )


def _answers(client: Any, model: str) -> bool:
    try:
        client.models.generate_content(model=model, contents=_PROBE_PROMPT)
        return True
    except Exception as e:
        logger.info("research_model_unavailable", model=model, error=str(e)[:160])
        return False


def resolve_research_models(client: Any | None = None) -> dict[str, str]:
    """Probe each tier's preference list once; cache and return {tier: model}.

    If nothing answers, the tier keeps its first preference so the failure surfaces
    in the node that uses it (with that node's normal fallback/error handling).
    """
    client = client or _default_client()
    for tier in ("deep", "fast"):
        prefs = candidates(tier)  # type: ignore[arg-type]
        chosen = next((m for m in prefs if _answers(client, m)), prefs[0] if prefs else "")
        _resolved[tier] = chosen
        logger.info("research_model_resolved", tier=tier, model=chosen, preferences=prefs)
    return dict(_resolved)


def reset_resolved_models() -> None:
    """Testing hook: forget probe results."""
    _resolved.clear()

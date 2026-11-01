"""
Superlab registry — maps organisation / affiliation text to a known AI lab.

Used by research sourcing (HF Daily Papers `organization`, arXiv comments) and
by the ranker's lab-tier bonus. Tier 1 = frontier industry labs, tier 2 = top
academic groups and other major research orgs.

Matching is whole-word and case-insensitive so "meta-learning" or "metadata"
never resolve to Meta. Edit `LABS` to add or re-tier an organisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Lab:
    name: str
    tier: int
    aliases: tuple[str, ...]


# Decision D4: superlabs get a boost, not an automatic win (manual picks get +3.0).
LAB_TIER_BONUS: dict[int, float] = {1: 1.5, 2: 0.5}

LABS: tuple[Lab, ...] = (
    # ── Tier 1: frontier labs ────────────────────────────────
    Lab(
        "Google DeepMind",
        1,
        ("google deepmind", "deepmind", "google research", "google brain", "google"),
    ),
    Lab("OpenAI", 1, ("openai",)),
    Lab("Anthropic", 1, ("anthropic",)),
    Lab(
        "Meta",
        1,
        (
            "meta ai",
            "meta fair",
            "FAIR",
            "meta platforms",
            "meta superintelligence",
            "facebook",
            "meta-llama",
            "metaresearch",
            "meta",
        ),
    ),
    Lab("Microsoft", 1, ("microsoft research", "microsoft")),
    Lab("NVIDIA", 1, ("nvidia",)),
    Lab("Apple", 1, ("apple",)),
    Lab("DeepSeek", 1, ("deepseek-ai", "deepseek")),
    Lab("Alibaba Qwen", 1, ("qwen", "alibaba", "alibaba-nlp", "damo academy")),
    Lab("Mistral AI", 1, ("mistral ai", "mistralai", "mistral")),
    Lab("xAI", 1, ("xAI", "x.ai")),
    # ── Tier 2: major research orgs & top academia ──────────
    Lab("ByteDance Seed", 2, ("bytedance", "bytedance seed")),
    Lab("Tencent", 2, ("tencent",)),
    Lab("Moonshot AI", 2, ("moonshot ai", "moonshotai", "kimi")),
    Lab("Zhipu AI", 2, ("zhipu", "thudm", "z.ai")),
    Lab("Amazon", 2, ("amazon", "aws ai")),
    Lab("IBM Research", 2, ("ibm research", "ibm")),
    Lab("Salesforce Research", 2, ("salesforce",)),
    Lab("Cohere", 2, ("cohere", "cohere for ai")),
    Lab("Hugging Face", 2, ("hugging face", "huggingface")),
    Lab("AI2", 2, ("allen institute for ai", "allenai", "AI2")),
    Lab("Stanford", 2, ("stanford",)),
    Lab("MIT", 2, ("MIT", "massachusetts institute of technology", "mit csail")),
    Lab("UC Berkeley", 2, ("uc berkeley", "berkeley", "BAIR")),
    Lab("CMU", 2, ("carnegie mellon", "CMU")),
    Lab("Princeton", 2, ("princeton",)),
    Lab("Tsinghua", 2, ("tsinghua",)),
    Lab("Mila", 2, ("Mila", "mila quebec")),
    Lab("Oxford", 2, ("university of oxford", "oxford")),
)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Whole-word pattern; a hyphen is not a boundary, so "meta-learning" is not Meta.

    Aliases containing uppercase letters match case-sensitively, so acronyms like
    "FAIR" or "MIT" don't fire on "a fair comparison". "MIT License" is excluded
    explicitly because arXiv comments often mention it.
    """
    flags = 0 if any(c.isupper() for c in alias) else re.IGNORECASE
    tail = r"(?!\s+(?i:licen[cs]e))" if alias == "MIT" else ""
    return re.compile(rf"(?<![\w-]){re.escape(alias)}(?![\w-]){tail}", flags)


_COMPILED: tuple[tuple[Lab, tuple[re.Pattern[str], ...]], ...] = tuple(
    (lab, tuple(_alias_pattern(a) for a in lab.aliases)) for lab in LABS
)


# Single-word aliases that are unambiguous even in free text (arXiv comments).
_UNAMBIGUOUS_SINGLE_WORDS = frozenset({"deepmind", "openai", "anthropic", "deepseek-ai", "nvidia"})


def _is_strict_alias(alias: str) -> bool:
    return " " in alias or alias.lower() in _UNAMBIGUOUS_SINGLE_WORDS


_COMPILED_STRICT: tuple[tuple[Lab, tuple[re.Pattern[str], ...]], ...] = tuple(
    (lab, tuple(_alias_pattern(a) for a in lab.aliases if _is_strict_alias(a))) for lab in LABS
)


def detect_lab(*texts: str | None, strict: bool = False) -> Lab | None:
    """Return the highest-tier lab mentioned in any of `texts` (registry order breaks ties).

    `strict=True` is for free text such as arXiv comments: only multi-word org names
    ("google deepmind", "microsoft research") and unambiguous names ("openai") count,
    so "built on Qwen" or "github.com/google/jax" do not award a lab bonus.
    """
    haystack = " | ".join(t for t in texts if t)
    if not haystack.strip():
        return None
    table = _COMPILED_STRICT if strict else _COMPILED
    found = [lab for lab, patterns in table if any(p.search(haystack) for p in patterns)]
    if not found:
        return None
    return min(found, key=lambda lab: lab.tier)


def lab_bonus(lab: Lab | None) -> float:
    return LAB_TIER_BONUS.get(lab.tier, 0.0) if lab else 0.0

"""
Text normalization for LLM-produced strings before they reach any renderer.

Handles the LaTeX-leak failure mode: arXiv titles and abstracts routinely
contain inline math (`$...$`), `\\text{...}` wrappers, and `^`/`_`
super/subscript markup that reads fine in a paper PDF but renders as raw
literal syntax ("M$^\\text{4}$World") when dropped into an HTML slide.

`normalize_text` is intentionally conservative about bare (unbraced)
`^`/`_` — it only rewrites digit exponents/subscripts outside braces, so
ordinary prose like `snake_case_identifier` is left untouched. Braced forms
(`^{...}`, `_{...}`) are LaTeX by construction and always rewritten.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_TEX_WRAPPER_RE = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|textbf|textit|emph)\{([^{}]*)\}"
)
_STRAY_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")
_SUPERSCRIPT_BRACED_RE = re.compile(r"\^\{([^{}]*)\}")
_SUPERSCRIPT_DIGIT_RE = re.compile(r"\^(\d)")
_SUBSCRIPT_BRACED_RE = re.compile(r"_\{([^{}]*)\}")
_SUBSCRIPT_DIGIT_RE = re.compile(r"_(\d)")
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")

_SUPERSCRIPT_MAP: dict[str, str] = {
    "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
    "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
    "+": "\u207a", "-": "\u207b", "n": "\u207f", "i": "\u2071",
}
_SUBSCRIPT_MAP: dict[str, str] = {
    "0": "\u2080", "1": "\u2081", "2": "\u2082", "3": "\u2083", "4": "\u2084",
    "5": "\u2085", "6": "\u2086", "7": "\u2087", "8": "\u2088", "9": "\u2089",
    "+": "\u208a", "-": "\u208b",
}


def _map_chars(chars: str, table: dict[str, str]) -> str:
    """Map each char via `table`; characters with no mapping pass through unchanged."""
    return "".join(table.get(c, c) for c in chars)


def normalize_text(text: str) -> str:
    """Strip LaTeX markup from LLM-produced text, mapping sub/superscripts to Unicode."""
    if not text:
        return text

    result = text
    prev = None
    while prev != result:
        prev = result
        result = _TEX_WRAPPER_RE.sub(r"\1", result)

    result = _SUPERSCRIPT_BRACED_RE.sub(lambda m: _map_chars(m.group(1), _SUPERSCRIPT_MAP), result)
    result = _SUPERSCRIPT_DIGIT_RE.sub(lambda m: _SUPERSCRIPT_MAP.get(m.group(1), m.group(1)), result)
    result = _SUBSCRIPT_BRACED_RE.sub(lambda m: _map_chars(m.group(1), _SUBSCRIPT_MAP), result)
    result = _SUBSCRIPT_DIGIT_RE.sub(lambda m: _SUBSCRIPT_MAP.get(m.group(1), m.group(1)), result)

    result = _STRAY_COMMAND_RE.sub(r"\1", result)
    result = result.replace("$", "")
    result = _WHITESPACE_RE.sub(" ", result).strip()
    return result


def normalize_title(title: str) -> str:
    """Alias of `normalize_text` for call-site clarity at title render points."""
    return normalize_text(title)


def normalize_model_strings(model: BaseModel) -> BaseModel:
    """Return a copy of `model` with `normalize_text` applied to every str field
    and every str item inside list fields. Non-string fields pass through untouched."""
    data = model.model_dump()
    normalized = {
        key: (
            normalize_text(value) if isinstance(value, str)
            else [normalize_text(item) if isinstance(item, str) else item for item in value]
            if isinstance(value, list)
            else value
        )
        for key, value in data.items()
    }
    return model.__class__(**normalized)

"""
Deterministic significance-verdict calibration.

`RichDeepAnalysis.significance_verdict` used to be assigned freely by the
same LLM call that wrote the rest of the analysis, decoupled from
`ResearchScores` (the gauge numbers shown on the cover slide). That let a
"Major Contribution" verdict sit next to a 1/10 reproducibility gauge —
self-discrediting and confusing. Verdict is now a pure function of the
gauge scores, computed once those scores exist, so the label and the
gauges can never contradict each other.
"""

from __future__ import annotations

_PARADIGM_SHIFT = "Paradigm Shift"
_MAJOR_CONTRIBUTION = "Major Contribution"
_SOLID_CONTRIBUTION = "Solid Contribution"
_INCREMENTAL = "Incremental"


def calibrate_verdict(scores: dict) -> str:
    """Return a significance verdict derived only from the 1-10 gauge scores.

    Thresholds (calibrated against the label semantics, not the LLM's judgment):
      Paradigm Shift     — benchmark_improvement >= 8 AND reproducibility >= 7 AND novelty >= 8
      Major Contribution — benchmark_improvement >= 7 AND reproducibility >= 5
      Solid Contribution — benchmark_improvement >= 5 OR novelty >= 6
      Incremental         — everything else
    """
    benchmark = scores.get("benchmark_improvement", 0)
    reproducibility = scores.get("reproducibility", 0)
    novelty = scores.get("novelty", 0)

    if benchmark >= 8 and reproducibility >= 7 and novelty >= 8:
        return _PARADIGM_SHIFT
    if benchmark >= 7 and reproducibility >= 5:
        return _MAJOR_CONTRIBUTION
    if benchmark >= 5 or novelty >= 6:
        return _SOLID_CONTRIBUTION
    return _INCREMENTAL

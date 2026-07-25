"""Shared SVG score-visual renderers (light theme, 1-10 scale).

Extracted from research_graph.py so research_carousel.py can reuse it without
creating a circular import (research_graph imports research_carousel_node).

Also home to the deck's role-based accent colour constants. The design spec
enforces exactly one colour per ROLE, not per slide-type:
  - ACCENT_CONTRIBUTION    — this paper's own contribution (hook-stat number,
                             core-approach/architecture diagram, any
                             quantified/verified result or score)
  - ACCENT_PRIOR_ART_GREY  — prior art / comparison / baseline content
  - ACCENT_UNVERIFIED_AMBER — unverified or missing claims (matches the shared
                             --amber CSS custom property in the slide template)
"""

from __future__ import annotations

import math

ACCENT_CONTRIBUTION = "#0EA5E9"      # Sky-500 — this paper's own contribution
ACCENT_PRIOR_ART_GREY = "#64748B"    # Slate-500 — prior art / comparison / baseline
ACCENT_UNVERIFIED_AMBER = "#D97706"  # Amber-600 — unverified or missing claim


def render_gauge_svg(label: str, value: int, color: str) -> str:
    """Return an inline SVG radial gauge for a 1-10 score (light theme)."""
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
        f' stroke="#E2E8F0" stroke-width="9" stroke-linecap="round"'
        f' stroke-dasharray="{bg_dash}" transform="{transform}"/>'
        f'<circle cx="60" cy="60" r="{r:.0f}" fill="none"'
        f' stroke="{color}" stroke-width="9" stroke-linecap="round"'
        f' stroke-dasharray="{fg_dash}" transform="{transform}"/>'
        f'<text x="60" y="55" text-anchor="middle" fill="#0F172A"'
        f' font-size="24" font-weight="700" font-family="JetBrains Mono,monospace">{value}</text>'
        f'<text x="60" y="72" text-anchor="middle" fill="#94A3B8"'
        f' font-size="11" font-family="JetBrains Mono,monospace">/10</text>'
        f'</svg>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        f'color:#64748B;text-transform:uppercase;letter-spacing:1.5px;'
        f'text-align:center;margin-top:-4px">{label}</div>'
        f'</div>'
    )


def render_bar_strip_svg(label: str, value: int, color: str) -> str:
    """Return an inline SVG horizontal bar segment for a 1-10 score (light theme).

    Drop-in replacement for `render_gauge_svg` — same (label, value, color)
    signature and return type — so existing call sites only need to swap the
    function reference. Four of these side by side read as a single low-height
    bar strip instead of four tall radial rings, cutting vertical space while
    staying legible at mobile width (label + numeric value inline, no need to
    infer magnitude from arc angle).
    """
    width = 220.0
    height = 40.0
    track_h = 6.0
    track_y = height - track_h
    fill_w = max(0.0, min(width, (value / 10) * width))

    return (
        f'<div style="display:flex;flex-direction:column;flex:1;min-width:0;gap:3px">'
        f'<svg width="100%" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}"'
        f' preserveAspectRatio="none">'
        f'<text x="0" y="14" fill="#64748B" font-size="10" font-weight="700"'
        f' font-family="JetBrains Mono,monospace" letter-spacing="1">{label.upper()}</text>'
        f'<text x="{width:.0f}" y="14" text-anchor="end" fill="{color}" font-size="14" font-weight="800"'
        f' font-family="JetBrains Mono,monospace">{value}'
        f'<tspan fill="#94A3B8" font-size="9" font-weight="400">/10</tspan></text>'
        f'<rect x="0" y="{track_y:.0f}" width="{width:.0f}" height="{track_h:.0f}" rx="3" fill="#E2E8F0"/>'
        f'<rect x="0" y="{track_y:.0f}" width="{fill_w:.2f}" height="{track_h:.0f}" rx="3" fill="{color}"/>'
        f'</svg>'
        f'</div>'
    )

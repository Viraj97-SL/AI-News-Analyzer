"""Shared SVG radial gauge renderer (light theme, 1-10 scale).

Extracted from research_graph.py so research_carousel.py can reuse it without
creating a circular import (research_graph imports research_carousel_node).
"""

from __future__ import annotations

import math


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

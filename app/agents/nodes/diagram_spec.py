"""
Structured architecture diagram generation.

Instead of relying solely on an extracted PDF/HTML figure (which may not
exist or may fail the figure-quality gate) or an ASCII-art fallback, the
LLM emits a small graph spec (nodes + edges) as JSON, which is rendered
server-side to inline SVG. This guarantees every paper gets a real
box-and-arrow diagram, not just papers with a usable extracted figure.

Only runs when no quality-passed extracted figure exists — see
`generate_architecture_spec_node`'s early-return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.llm_retry import invoke_with_schema_retry
from app.agents.nodes.text_utils import normalize_text
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DiagramNode(BaseModel):
    id: str = Field(description="Short unique lowercase slug, e.g. 'encoder'")
    label: str = Field(description="Short display label, max ~28 chars, e.g. 'Vision Encoder'")
    kind: Literal["input", "process", "output"] = Field(
        description="input=data source, process=model component, output=result"
    )


class DiagramEdge(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    label: str = Field(default="", description="Short edge label, optional")


class ArchitectureDiagramSpec(BaseModel):
    nodes: list[DiagramNode] = Field(
        description="4-8 pipeline stages/components, in left-to-right data-flow order"
    )
    edges: list[DiagramEdge] = Field(
        description="Directed edges connecting node ids; must reference only ids present in nodes"
    )


_KIND_COLORS = {"input": "#64748B", "process": "#0EA5E9", "output": "#059669"}
_NODE_W, _NODE_H, _H_GAP, _V_GAP, _MARGIN = 150, 56, 60, 40, 20


def render_diagram_svg(spec: ArchitectureDiagramSpec, max_width: int = 720) -> str:
    """Render a left-to-right box-and-arrow diagram as inline SVG, wrapping to
    additional rows when the node count would overflow `max_width`."""
    if not spec.nodes:
        return ""

    valid_ids = {n.id for n in spec.nodes}
    edges = [e for e in spec.edges if e.source in valid_ids and e.target in valid_ids]

    per_row = max(1, (max_width - _MARGIN) // (_NODE_W + _H_GAP))
    positions: dict[str, tuple[int, int]] = {}
    for i, node in enumerate(spec.nodes):
        row, col = divmod(i, per_row)
        x = _MARGIN + col * (_NODE_W + _H_GAP)
        y = _MARGIN + row * (_NODE_H + _V_GAP)
        positions[node.id] = (x, y)

    rows = (len(spec.nodes) - 1) // per_row + 1
    cols_in_widest_row = min(len(spec.nodes), per_row)
    width = _MARGIN * 2 + cols_in_widest_row * _NODE_W + max(0, cols_in_widest_row - 1) * _H_GAP
    height = _MARGIN * 2 + rows * _NODE_H + max(0, rows - 1) * _V_GAP

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">',
        '<defs><marker id="diagram-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#94A3B8"/></marker></defs>',
    ]

    for edge in edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        x1c, y1c = x1 + _NODE_W, y1 + _NODE_H / 2
        x2c, y2c = x2, y2 + _NODE_H / 2
        parts.append(
            f'<line x1="{x1c}" y1="{y1c}" x2="{x2c}" y2="{y2c}" '
            f'stroke="#94A3B8" stroke-width="2" marker-end="url(#diagram-arrow)"/>'
        )
        if edge.label:
            mx, my = (x1c + x2c) / 2, (y1c + y2c) / 2 - 6
            parts.append(
                f'<text x="{mx}" y="{my}" font-size="10" fill="#64748B" text-anchor="middle">{edge.label}</text>'
            )

    for node in spec.nodes:
        x, y = positions[node.id]
        color = _KIND_COLORS.get(node.kind, "#0EA5E9")
        parts.append(
            f'<rect x="{x}" y="{y}" width="{_NODE_W}" height="{_NODE_H}" rx="8" '
            f'fill="white" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + _NODE_W / 2}" y="{y + _NODE_H / 2 + 4}" font-size="12" '
            f'fill="#0F172A" text-anchor="middle">{node.label[:24]}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def generate_architecture_spec_node(state: "PipelineState") -> dict:
    """Generate a guaranteed structured diagram — skipped when a real extracted
    figure already passed the quality gate (that's strictly more informative)."""
    if (state.get("architecture_diagram_b64") or "").strip():
        return {"architecture_spec_svg": "", "current_step": "architecture_spec_not_needed"}

    analysis = state.get("deep_analysis", {})
    methodology = analysis.get("methodology", "")
    technical_innovation = analysis.get("technical_innovation", "")
    if not methodology:
        return {"architecture_spec_svg": "", "current_step": "architecture_spec_skipped"}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            api_key=settings.google_api_key,
        ).with_structured_output(ArchitectureDiagramSpec)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract the paper's architecture/pipeline as a diagram spec: 4-8 nodes "
             "(stages or components, in left-to-right data-flow order) and the edges "
             "connecting them. Node ids must be short lowercase slugs. Every edge must "
             "reference node ids that exist in your nodes list."),
            ("user", "Methodology:\n{methodology}\n\nWhat's new:\n{technical_innovation}"),
        ])

        spec: ArchitectureDiagramSpec = invoke_with_schema_retry(
            lambda: (prompt | llm).invoke({
                "methodology": methodology,
                "technical_innovation": technical_innovation,
            }),
            max_retries=settings.llm_schema_max_retries,
            logger=logger,
            context="architecture_spec_generation",
        )
    except Exception as e:
        logger.warning("architecture_spec_generation_failed", error=str(e))
        return {"architecture_spec_svg": "", "current_step": "architecture_spec_skipped"}

    if not spec.nodes:
        return {"architecture_spec_svg": "", "current_step": "architecture_spec_skipped"}

    spec = spec.model_copy(update={
        "nodes": [n.model_copy(update={"label": normalize_text(n.label)}) for n in spec.nodes],
        "edges": [e.model_copy(update={"label": normalize_text(e.label)}) for e in spec.edges],
    })

    svg = render_diagram_svg(spec)
    logger.info("architecture_spec_generated", nodes=len(spec.nodes), edges=len(spec.edges))
    return {"architecture_spec_svg": svg, "current_step": "architecture_spec_generated"}

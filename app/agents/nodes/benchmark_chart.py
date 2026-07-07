"""
Benchmark chart node — extracts numeric metrics from paper breakthroughs and
renders a light-themed chart PNG: a horizontal bar chart for 2+ metrics, or a
single stat card when there's exactly one metric with no prior-SOTA value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")


class BenchmarkMetric(BaseModel):
    metric_name: str = Field(description="e.g. 'MMLU', 'HumanEval', 'WMT-23 BLEU'")
    new_paper_value: float = Field(description="The new paper's score/number")
    prior_sota_value: float | None = Field(
        default=None, description="Prior SOTA value if mentioned, else null"
    )
    unit: str = Field(default="", description="e.g. '%', 'points', 'BLEU score', or empty string")


class BenchmarkExtraction(BaseModel):
    metrics: list[BenchmarkMetric] = Field(description="Up to 5 key benchmark metrics. Empty list if no clear numbers exist.")


def benchmark_chart_node(state: "PipelineState") -> dict:
    """Extract benchmark metrics via LLM and render a dark-themed bar chart PNG."""
    analysis = state.get("deep_analysis", {})
    run_id = state.get("run_id", "dev")

    breakthroughs = analysis.get("breakthroughs", "")
    if not breakthroughs:
        return {"benchmark_metrics": [], "benchmark_chart_path": "", "current_step": "benchmark_skipped"}

    # ── 1. Extract metrics ────────────────────────────────────────────────
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            api_key=settings.google_api_key,
        ).with_structured_output(BenchmarkExtraction)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract concrete benchmark metrics (numbers, scores, percentages) from the text. "
             "Include prior SOTA values if mentioned. Return up to 5 metrics. "
             "If no clear numeric benchmarks exist, return an empty metrics list."),
            ("user", "{breakthroughs}"),
        ])

        extraction: BenchmarkExtraction = (prompt | llm).invoke({"breakthroughs": breakthroughs})
    except Exception as e:
        logger.warning("benchmark_extraction_failed", error=str(e))
        return {"benchmark_metrics": [], "benchmark_chart_path": "", "current_step": "benchmark_skipped"}

    if not extraction.metrics:
        logger.info("no_benchmark_metrics_found")
        return {"benchmark_metrics": [], "benchmark_chart_path": "", "current_step": "benchmark_skipped"}

    # ── 2. Render chart (stat card for a lone metric, bar chart otherwise) ─
    try:
        metrics = extraction.metrics[:5]
        n = len(metrics)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = str(OUTPUT_DIR / f"benchmark_chart_{run_id}.png")

        if n == 1 and metrics[0].prior_sota_value is None:
            _render_stat_card(metrics[0], chart_path)
        else:
            _render_bar_chart(metrics, chart_path)

        logger.info("benchmark_chart_generated", path=chart_path, metric_count=n)
        return {
            "benchmark_metrics": [m.model_dump() for m in metrics],
            "benchmark_chart_path": chart_path,
            "current_step": "benchmark_chart_generated",
        }

    except ImportError:
        logger.warning("matplotlib_not_installed", hint="pip install matplotlib")
        return {
            "benchmark_metrics": [m.model_dump() for m in extraction.metrics],
            "benchmark_chart_path": "",
            "current_step": "benchmark_skipped",
        }
    except Exception as e:
        logger.error("benchmark_chart_render_failed", error=str(e))
        return {"benchmark_metrics": [], "benchmark_chart_path": "", "current_step": "benchmark_skipped"}


def _render_stat_card(metric: BenchmarkMetric, chart_path: str) -> None:
    """Render a single large centered stat — used when there's exactly one metric
    and no prior-SOTA value, since a bar chart degenerates to an unreadable sliver."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value_str = f"{metric.new_paper_value:g}"
    if metric.unit == "%":
        value_str += "%"
    elif metric.unit:
        value_str += f" {metric.unit}"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.axis("off")

    ax.text(
        0.5, 0.60, value_str, transform=ax.transAxes, ha="center", va="center",
        fontsize=64, fontweight="bold", color="#0EA5E9", fontfamily="monospace",
    )
    ax.text(
        0.5, 0.30, metric.metric_name, transform=ax.transAxes, ha="center", va="center",
        fontsize=20, color="#334155", fontfamily="monospace",
    )
    ax.text(
        0.5, 0.15, "THIS PAPER", transform=ax.transAxes, ha="center", va="center",
        fontsize=11, color="#94A3B8", fontfamily="monospace", fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(chart_path, dpi=200, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)


def _render_bar_chart(metrics: list[BenchmarkMetric], chart_path: str) -> None:
    """Render a horizontal bar chart, sized and scaled for readability whether
    there are 1-2 metrics (squarer, capped bar footprint) or 3-5 (original layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    labels = [
        f"{m.metric_name} ({m.unit})" if m.unit else m.metric_name
        for m in metrics
    ]
    new_vals = [m.new_paper_value for m in metrics]
    sota_vals = [m.prior_sota_value for m in metrics]
    has_sota = any(v is not None for v in sota_vals)

    n = len(metrics)
    small_n = n <= 2
    label_fs = 13 if small_n else 9
    tick_fs = 11 if small_n else 8
    title_fs = 14 if small_n else 11
    value_fs = 14 if small_n else 9
    dpi = 200 if small_n else 150

    if small_n:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig_height = max(2.8, n * 0.95 + (0.7 if has_sota else 0))
        fig, ax = plt.subplots(figsize=(9, fig_height))

    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    bar_h = 0.35
    y_pos = list(range(n))

    if has_sota:
        ax.barh(
            [y + bar_h / 2 for y in y_pos], new_vals, bar_h,
            color="#0EA5E9", label="This Paper", zorder=3,
        )
        for i, (y, v) in enumerate(zip(y_pos, sota_vals, strict=True)):
            if v is not None:
                ax.barh(
                    y - bar_h / 2, v, bar_h,
                    color="#CBD5E1", label="Prior SOTA" if i == 0 else "",
                    zorder=3,
                )
    else:
        ax.barh(y_pos, new_vals, 0.55, color="#0EA5E9", zorder=3)

    if small_n:
        # Cap whitespace around 1-2 bars so they don't shrink to a sliver.
        ax.set_ylim(-0.6, n - 0.4)

    # Value annotations
    max_val = max(new_vals) if new_vals else 1
    for y, v in zip(y_pos, new_vals, strict=True):
        offset = bar_h / 2 if has_sota else 0
        ax.text(
            v + max_val * 0.01, y + offset, f"{v}",
            va="center", color="#0F172A", fontsize=value_fs,
            fontfamily="monospace", fontweight="bold",
        )

    # Styling — matplotlib uses (R,G,B,A) tuples, not CSS rgba()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=label_fs, color="#334155", fontfamily="monospace")
    ax.tick_params(colors="#475569", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor((0.88, 0.91, 0.94, 1.0))  # Slate-200
    ax.tick_params(axis="x", colors=(0.45, 0.51, 0.59, 1.0), labelsize=tick_fs)  # Slate-500
    ax.set_xlabel("Score", color=(0.45, 0.51, 0.59, 1.0), fontfamily="monospace", fontsize=tick_fs)
    ax.set_title(
        "Benchmark Comparison",
        color="#0EA5E9", fontfamily="monospace", fontsize=title_fs, pad=10, fontweight="bold",
    )
    ax.grid(axis="x", color=(0.88, 0.91, 0.94, 1.0), linewidth=0.8, zorder=0)

    if has_sota:
        blue_p = mpatches.Patch(color="#0EA5E9", label="This Paper")
        gray_p = mpatches.Patch(color="#CBD5E1", label="Prior SOTA")
        ax.legend(
            handles=[blue_p, gray_p],
            facecolor="#F8FAFC", edgecolor=(0.88, 0.91, 0.94, 1.0),
            labelcolor="#334155", fontsize=tick_fs, prop={"family": "monospace"},
        )

    plt.tight_layout(pad=0.5)
    plt.savefig(chart_path, dpi=dpi, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)

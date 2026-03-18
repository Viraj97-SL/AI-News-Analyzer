"""
Benchmark chart node — extracts numeric metrics from paper breakthroughs
and renders a dark-themed horizontal bar chart PNG.
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

    # ── 2. Render bar chart ───────────────────────────────────────────────
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches  # type: ignore[import]
        import matplotlib.pyplot as plt  # type: ignore[import]

        metrics = extraction.metrics[:5]
        labels = [
            f"{m.metric_name} ({m.unit})" if m.unit else m.metric_name
            for m in metrics
        ]
        new_vals = [m.new_paper_value for m in metrics]
        sota_vals = [m.prior_sota_value for m in metrics]
        has_sota = any(v is not None for v in sota_vals)

        n = len(metrics)
        fig_height = max(2.8, n * 0.95 + (0.7 if has_sota else 0))
        fig, ax = plt.subplots(figsize=(9, fig_height))

        fig.patch.set_facecolor("#030305")
        ax.set_facecolor("#0a0a14")

        bar_h = 0.35
        y_pos = list(range(n))

        if has_sota:
            ax.barh(
                [y + bar_h / 2 for y in y_pos], new_vals, bar_h,
                color="#00f3ff", label="This Paper", zorder=3,
            )
            for i, (y, v) in enumerate(zip(y_pos, sota_vals)):
                if v is not None:
                    ax.barh(
                        y - bar_h / 2, v, bar_h,
                        color="#445566", label="Prior SOTA" if i == 0 else "",
                        zorder=3,
                    )
        else:
            ax.barh(y_pos, new_vals, 0.55, color="#00f3ff", zorder=3)

        # Value annotations
        max_val = max(new_vals) if new_vals else 1
        for i, (y, v) in enumerate(zip(y_pos, new_vals)):
            offset = bar_h / 2 if has_sota else 0
            ax.text(
                v + max_val * 0.01, y + offset, f"{v}",
                va="center", color="white", fontsize=9,
                fontfamily="monospace", fontweight="bold",
            )

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, color="#ccccdd", fontfamily="monospace")
        ax.tick_params(colors="white", length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("rgba(0, 243, 255, 0.2)")
        ax.tick_params(axis="x", colors="rgba(255,255,255,0.4)", labelsize=8)
        ax.set_xlabel("Score", color="rgba(255,255,255,0.4)", fontfamily="monospace", fontsize=9)
        ax.set_title(
            "Benchmark Comparison",
            color="#00f3ff", fontfamily="monospace", fontsize=11, pad=10, fontweight="bold",
        )
        ax.grid(axis="x", color="rgba(0, 243, 255, 0.1)", linewidth=0.5, zorder=0)

        if has_sota:
            cyan_p = mpatches.Patch(color="#00f3ff", label="This Paper")
            gray_p = mpatches.Patch(color="#445566", label="Prior SOTA")
            ax.legend(
                handles=[cyan_p, gray_p],
                facecolor="#0a0a14", edgecolor="rgba(0,243,255,0.3)",
                labelcolor="white", fontsize=8, prop={"family": "monospace"},
            )

        plt.tight_layout(pad=0.5)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = str(OUTPUT_DIR / f"benchmark_chart_{run_id}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#030305")
        plt.close(fig)

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

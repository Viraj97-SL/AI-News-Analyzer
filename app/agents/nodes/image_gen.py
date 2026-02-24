"""
Image generation node — branded news card images via html2image.

Produces 1200x627px cards (LinkedIn optimal) using HTML/CSS templates
rendered through headless Chrome. Zero API cost, full design control.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


def image_gen_node(state: PipelineState) -> dict:
    """Generate branded news card images for top stories."""
    summaries = state.get("summaries", [])
    if not summaries:
        logger.info("image_gen_skipped", reason="no summaries available")
        return {"image_paths": [], "current_step": "images_generated"}

    try:
        from html2image import Html2Image

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        hti = Html2Image(
            output_path=str(OUTPUT_DIR),
            size=(1200, 627),
            custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
        )

        # Load Jinja2 template
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("news_card.html")

        image_paths: list[str] = []
        for i, summary in enumerate(summaries[:5]):  # max 5 cards per run
            html = template.render(
                headline=summary.get("headline", "AI News Update"),
                body=summary.get("body", "")[:180],
                category=summary.get("category", "AI"),
                source_count=len(summary.get("source_urls", [])),
                credibility=f"{summary.get('credibility_score', 0):.0%}",
                run_id=state.get("run_id", ""),
            )

            filename = f"card_{state.get('run_id', 'dev')}_{i}.png"
            hti.screenshot(html_str=html, save_as=filename)
            image_paths.append(str(OUTPUT_DIR / filename))

        logger.info("images_generated", count=len(image_paths))
        return {"image_paths": image_paths, "current_step": "images_generated"}

    except ImportError:
        logger.warning("html2image_not_installed", hint="pip install html2image")
        return {"image_paths": [], "error_log": ["html2image not installed"]}
    except Exception as e:
        logger.error("image_gen_error", error=str(e))
        return {"image_paths": [], "error_log": [f"Image gen error: {e}"]}

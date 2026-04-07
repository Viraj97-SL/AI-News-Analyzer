"""
Image generation node — news cards + LinkedIn carousel PDF via html2image.

Produces:
  - 1200x627px individual news cards (used in email newsletter)
  - 1080x1080px carousel slides combined into a PDF (LinkedIn document post)

Carousel slide order:
  1. Cover  — story count + category breakdown chips
  2. Snapshot — "Week in Numbers" infographic (4 stat boxes + bar chart)
  3. Story × N — headline + bullet points + key stat callout + credibility bar
  4. Closing — CTA + follow prompt
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
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

# Color palette for categories (matches template CSS vars)
CATEGORY_COLORS: dict[str, str] = {
    "LLM":             "#00f3ff",
    "Computer Vision": "#ffe600",
    "Robotics":        "#ff2d78",
    "AI Policy":       "#ff6b35",
    "AI Startup":      "#00ff9d",
    "Research":        "#9d00ff",
    "Industry":        "#7ec8ff",
    "Other":           "#888888",
}

# Patterns to extract a key stat/number from a body string.
# Listed in priority order — first match wins.
_STAT_PATTERNS: list[tuple[str, str]] = [
    # Dollar amounts:  $4.6 billion,  $200M,  $1.2T
    (r"\$[\d.,]+\s*(?:billion|trillion|million|B|M|T)\b", "funding / valuation"),
    # Percentages:  94.2%,  38%
    (r"\b\d+(?:\.\d+)?\s*%", "improvement / accuracy"),
    # Multipliers:  3× faster,  10x better
    (r"\b\d+(?:\.\d+)?\s*[x×]\s*(?:faster|better|more|improvement|speedup)\b", "speedup"),
    # AI parameter counts:  70B parameters,  405B tokens
    (r"\b\d+(?:\.\d+)?[BM]\b(?=\s*(?:param|token|model))", "parameters"),
    # Benchmark scores with numbers:  top 3,  #1
    (r"(?:top|#)\s*\d+\b", "ranking"),
    # Large plain numbers:  42 billion,  200 million
    (r"\b\d+(?:\.\d+)?\s*(?:billion|million)\b", "scale"),
]


def _extract_key_stat(body: str) -> dict | None:
    """
    Find the most prominent number/statistic in a summary body.

    Returns a dict with 'value' (the raw matched string, trimmed) and
    'label' (2–4 words of surrounding context), or None if nothing found.
    """
    for pattern, fallback_label in _STAT_PATTERNS:
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            # Build a short label from the words that follow the match
            after = body[m.end():].strip()
            label_words = after.split()[:4]
            label = " ".join(label_words).rstrip(".,;:") or fallback_label
            return {"value": m.group(0).strip(), "label": label}
    return None


def _extract_key_points(body: str) -> list[str]:
    """Split a summary body into up to 4 readable bullet points."""
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    return [s for s in sentences if len(s) > 20][:4]


def _build_category_breakdown(summaries: list[dict]) -> list[dict]:
    """Count stories per category, return sorted list with colour + percentage."""
    counts = Counter(s.get("category", "Other") for s in summaries)
    total = len(summaries)
    return [
        {
            "name": cat,
            "count": count,
            "pct": round(count / total * 100) if total else 0,
            "color": CATEGORY_COLORS.get(cat, CATEGORY_COLORS["Other"]),
        }
        for cat, count in counts.most_common()
    ]


def _build_week_stats(summaries: list[dict]) -> dict:
    """Aggregate key numbers shown on the 'Week in Numbers' snapshot slide."""
    all_sources: set[str] = set()
    for s in summaries:
        all_sources.update(s.get("source_urls", []))

    scores = [s.get("credibility_score", 0.0) for s in summaries]
    avg_cred = sum(scores) / len(scores) if scores else 0.0
    categories = {s.get("category", "Other") for s in summaries}

    return {
        "story_count": len(summaries),
        "source_count": len(all_sources) or sum(len(s.get("source_urls", [])) for s in summaries),
        "avg_credibility": f"{avg_cred:.0%}",
        "category_count": len(categories),
    }


def _make_hti(size: tuple[int, int]):
    from html2image import Html2Image

    return Html2Image(
        output_path=str(OUTPUT_DIR),
        size=size,
        custom_flags=["--no-sandbox", "--hide-scrollbars", "--disable-gpu"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Email cards (1200 × 627)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_cards(summaries: list[dict], run_id: str, env: Environment) -> list[str]:
    """Render individual 1200×627 news cards for email attachments."""
    template = env.get_template("news_card.html")
    hti = _make_hti((1200, 627))
    paths: list[str] = []

    for i, summary in enumerate(summaries[:5]):
        html = template.render(
            headline=summary.get("headline", "AI News Update"),
            body=summary.get("body", "")[:180],
            category=summary.get("category", "AI"),
            source_count=len(summary.get("source_urls", [])),
            credibility=f"{summary.get('credibility_score', 0):.0%}",
            run_id=run_id,
        )
        filename = f"card_{run_id}_{i}.png"
        hti.screenshot(html_str=html, save_as=filename)
        paths.append(str(OUTPUT_DIR / filename))

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# LinkedIn carousel (1080 × 1080 slides → PDF)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_carousel_pdf(summaries: list[dict], run_id: str, env: Environment) -> str | None:
    """
    Render infographic carousel slides and combine into a single PDF.

    Slide order:
      1  Cover          — story count + category breakdown chips
      2  Snapshot       — "Week in Numbers" 4-box infographic + bar chart
      3… Story slides   — headline, bullets, key stat callout, credibility bar
      N  Closing        — CTA + follow prompt
    """
    template = env.get_template("carousel_slide.html")
    hti = _make_hti((1080, 1080))

    story_summaries = summaries[:10]
    total_slides = len(story_summaries)
    date_str = date.today().strftime("%B %d, %Y")
    categories = _build_category_breakdown(story_summaries)
    stats = _build_week_stats(story_summaries)
    slide_pngs: list[str] = []

    # ── 1. Cover slide ────────────────────────────────────────
    html = template.render(
        slide_type="cover",
        story_count=total_slides,
        date_str=date_str,
        categories=categories,
    )
    name = f"carousel_{run_id}_cover.png"
    hti.screenshot(html_str=html, save_as=name)
    slide_pngs.append(str(OUTPUT_DIR / name))

    # ── 2. Snapshot "Week in Numbers" slide ───────────────────
    html = template.render(
        slide_type="snapshot",
        stats=stats,
        categories=categories,
    )
    name = f"carousel_{run_id}_snapshot.png"
    hti.screenshot(html_str=html, save_as=name)
    slide_pngs.append(str(OUTPUT_DIR / name))

    # ── 3. One story slide per summary ────────────────────────
    for i, summary in enumerate(story_summaries):
        body = summary.get("body", "")
        cred_score = summary.get("credibility_score", 0.0)
        source_urls = summary.get("source_urls", [])

        html = template.render(
            slide_type="story",
            slide_num=i + 1,
            total_slides=total_slides,
            headline=summary.get("headline", ""),
            key_points=_extract_key_points(body),
            key_stat=_extract_key_stat(body),
            category=summary.get("category", "AI"),
            credibility=f"{cred_score:.0%}",
            cred_pct=round(cred_score * 100),
            source_count=len(source_urls),
        )
        name = f"carousel_{run_id}_{i}.png"
        hti.screenshot(html_str=html, save_as=name)
        slide_pngs.append(str(OUTPUT_DIR / name))

    # ── 4. Closing / CTA slide ────────────────────────────────
    html = template.render(slide_type="closing")
    name = f"carousel_{run_id}_close.png"
    hti.screenshot(html_str=html, save_as=name)
    slide_pngs.append(str(OUTPUT_DIR / name))

    # ── Combine PNGs → PDF ────────────────────────────────────
    # Only include slides that were actually written to disk
    existing_pngs = [p for p in slide_pngs if Path(p).exists()]
    missing = len(slide_pngs) - len(existing_pngs)
    if missing:
        logger.warning("carousel_slides_missing", missing=missing, total=len(slide_pngs))

    if not existing_pngs:
        logger.error("carousel_no_slides_rendered")
        return None

    # Use PyMuPDF (fitz) — bundled codecs, no libjpeg/openjpeg dependency.
    import fitz  # type: ignore[import]

    pdf_path = str(OUTPUT_DIR / f"carousel_{run_id}.pdf")
    doc = fitz.open()
    for png_path in existing_pngs:
        img_doc = fitz.open(png_path)
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        img_pdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(img_pdf)
        img_pdf.close()
    doc.save(pdf_path)
    doc.close()

    logger.info("carousel_generated", slides=len(existing_pngs), pdf=pdf_path)
    return pdf_path, existing_pngs


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ──────────────────────────────────────────────────────────────────────────────

def image_gen_node(state: PipelineState) -> dict:
    """Generate email cards and a LinkedIn carousel PDF from pipeline summaries."""
    summaries = state.get("summaries", [])
    if not summaries:
        logger.info("image_gen_skipped", reason="no summaries available")
        return {"image_paths": [], "current_step": "images_generated"}

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )

        run_id = state.get("run_id", "dev")

        image_paths = _generate_cards(summaries, run_id, env)
        logger.info("news_cards_generated", count=len(image_paths))

        result: dict = {"image_paths": image_paths, "current_step": "images_generated"}

        # Carousel is generated separately so its failure never discards the
        # news cards that were already produced above.
        try:
            carousel_result = _generate_carousel_pdf(summaries, run_id, env)
            if carousel_result:
                pdf_path, slide_paths = carousel_result
                result["carousel_pdf_path"] = pdf_path
                result["carousel_slide_paths"] = slide_paths
        except Exception as ce:
            logger.error("carousel_gen_error", error=str(ce))

        return result

    except ImportError:
        logger.warning("html2image_not_installed", hint="pip install html2image")
        return {"image_paths": [], "error_log": ["html2image not installed"]}
    except Exception as e:
        logger.error("image_gen_error", error=str(e))
        return {"image_paths": [], "error_log": [f"Image gen error: {e}"]}

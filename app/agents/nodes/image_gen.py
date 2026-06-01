"""
Image generation node — news cards + LinkedIn carousel PDF via html2image.

Produces:
  - 1200x627px individual news cards (used in email newsletter)
  - 1080x1080px carousel slides combined into a PDF (LinkedIn document post)

Carousel slide order:
  1. Cover  — story count + date + author branding
  2. Story × N — headline + 3 bullet points + real article image + trust badge
  3. Closing — CTA + follow prompt
"""

from __future__ import annotations

import base64
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
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
    (r"\$[\d.,]+\s*(?:billion|trillion|million|B|M|T)\b", "funding / valuation"),
    (r"\b\d+(?:\.\d+)?\s*%", "improvement / accuracy"),
    (r"\b\d+(?:\.\d+)?\s*[x×]\s*(?:faster|better|more|improvement|speedup)\b", "speedup"),
    (r"\b\d+(?:\.\d+)?[BM]\b(?=\s*(?:param|token|model))", "parameters"),
    (r"(?:top|#)\s*\d+\b", "ranking"),
    (r"\b\d+(?:\.\d+)?\s*(?:billion|million)\b", "scale"),
]


def _extract_key_stat(body: str) -> dict | None:
    """Find the most prominent number/statistic in a summary body."""
    for pattern, fallback_label in _STAT_PATTERNS:
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            after = body[m.end():].strip()
            label_words = after.split()[:4]
            label = " ".join(label_words).rstrip(".,;:") or fallback_label
            return {"value": m.group(0).strip(), "label": label}
    return None


def _extract_key_points(body: str) -> list[str]:
    """
    Split a summary body into up to 3 bullet points that always end at a natural
    boundary (sentence end, comma, or word boundary) — never mid-word or mid-idea.
    """
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    points: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        if len(s) <= 145:
            points.append(s)
        else:
            chunk = s[:145]
            # Prefer ending at a sentence-boundary punctuation within the chunk
            cut = -1
            for punct in [". ", ", ", "; "]:
                idx = chunk.rfind(punct)
                if idx > 60:
                    cut = idx + len(punct) - 1
                    break
            if cut == -1:
                # Fall back to last word boundary
                cut = chunk.rfind(" ")
            points.append(s[: cut].rstrip(" .,") + "…" if cut > 60 else chunk.rstrip() + "…")
        if len(points) == 3:
            break
    return points


def _credibility_tier(score: float, source_count: int) -> dict:
    """Map internal credibility score to an honest, display-safe trust badge."""
    if score >= 0.75:
        return {"label": "Major Outlets", "color": "#00ff9d", "icon": "◉"}
    elif score >= 0.55:
        return {"label": "Press Coverage", "color": "#ffe600", "icon": "◎"}
    else:
        return {"label": "Emerging Story", "color": "#888888", "icon": "○"}


def _download_image_data_uri(image_url: str) -> str | None:
    """Download an image URL and return a data URI (mime + base64). Returns None on failure."""
    try:
        with httpx.Client(timeout=8, follow_redirects=True) as client:
            resp = client.get(image_url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "").split(";")[0].strip()
            if not mime.startswith("image/"):
                return None
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.debug("image_download_failed", url=image_url, error=str(e))
        return None


def _fetch_og_image(url: str) -> str | None:
    """
    Fetch the og:image from an article page and return a data URI.
    Returns None on any failure.
    """
    try:
        with httpx.Client(timeout=5, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            html = resp.text

        # property then content
        og_match = re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if not og_match:
            # content then property
            og_match = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                html,
                re.IGNORECASE,
            )
        if not og_match:
            return None

        image_url = og_match.group(1).strip()
        if not image_url.startswith("http"):
            return None

        return _download_image_data_uri(image_url)

    except Exception as e:
        logger.debug("og_image_fetch_failed", url=url, error=str(e))
        return None


def _serper_image_search(headline: str) -> str | None:
    """
    Fallback: search Google Images via Serper for a relevant photo.
    Tries each result until one downloads successfully.
    """
    if not settings.serper_api_key or settings.serper_api_key.startswith("your-"):
        return None
    try:
        with httpx.Client(timeout=6) as client:
            resp = client.post(
                "https://google.serper.dev/images",
                headers={"X-API-KEY": settings.serper_api_key},
                json={"q": headline, "num": 5},
            )
            resp.raise_for_status()
            images = resp.json().get("images", [])

        for img in images:
            image_url = img.get("imageUrl", "")
            if not image_url or not image_url.startswith("http"):
                continue
            data_uri = _download_image_data_uri(image_url)
            if data_uri:
                logger.debug("serper_image_found", headline=headline[:50])
                return data_uri

    except Exception as e:
        logger.debug("serper_image_search_failed", headline=headline[:50], error=str(e))
    return None


def _fetch_story_image(source_url: str, headline: str) -> str | None:
    """
    Get a relevant image for a story slide.

    1. OG image from the article URL (editorial photo chosen by the publisher)
    2. Serper Google Images search for the story headline
    3. Returns None → template renders the category gradient fallback
    """
    if source_url:
        data_uri = _fetch_og_image(source_url)
        if data_uri:
            return data_uri

    return _serper_image_search(headline)


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

def _generate_carousel_pdf(summaries: list[dict], run_id: str, env: Environment) -> tuple[str, list[str]] | None:
    """
    Render infographic carousel slides and combine into a single PDF.

    Slide order:
      1  Cover     — story count + date + author branding
      2… Stories   — headline, 3 bullets, real article photo, trust badge
      N  Closing   — CTA + follow prompt
    """
    template = env.get_template("carousel_slide.html")
    hti = _make_hti((1080, 1080))

    story_summaries = summaries[:8]
    total_slides = len(story_summaries)
    date_str = date.today().strftime("%B %d, %Y")
    slide_pngs: list[str] = []

    # ── 1. Cover slide ────────────────────────────────────────
    html = template.render(
        slide_type="cover",
        story_count=total_slides,
        date_str=date_str,
    )
    name = f"carousel_{run_id}_cover.png"
    hti.screenshot(html_str=html, save_as=name)
    slide_pngs.append(str(OUTPUT_DIR / name))

    # ── 2. One story slide per summary ────────────────────────
    for i, summary in enumerate(story_summaries):
        body = summary.get("body", "")
        cred_score = summary.get("credibility_score", 0.0)
        source_urls = summary.get("source_urls", [])
        source_count = len(source_urls)
        category = summary.get("category", "Other")

        # Fetch real article image — OG photo first, Serper fallback
        primary_url = source_urls[0] if source_urls else ""
        headline = summary.get("headline", "")
        story_image_uri = _fetch_story_image(primary_url, headline)
        if story_image_uri:
            logger.debug("story_image_fetched", slide=i)
        else:
            logger.debug("story_image_fallback", slide=i)

        html = template.render(
            slide_type="story",
            slide_num=i + 1,
            total_slides=total_slides,
            headline=headline,
            key_points=_extract_key_points(body),
            key_stat=_extract_key_stat(body),
            category=category,
            category_color=CATEGORY_COLORS.get(category, CATEGORY_COLORS["Other"]),
            story_image_uri=story_image_uri,
            trust_tier=_credibility_tier(cred_score, source_count),
            source_count=source_count,
        )
        name = f"carousel_{run_id}_{i}.png"
        hti.screenshot(html_str=html, save_as=name)
        slide_pngs.append(str(OUTPUT_DIR / name))

    # ── 3. Closing / CTA slide ────────────────────────────────
    html = template.render(slide_type="closing")
    name = f"carousel_{run_id}_close.png"
    hti.screenshot(html_str=html, save_as=name)
    slide_pngs.append(str(OUTPUT_DIR / name))

    # ── Combine PNGs → PDF ────────────────────────────────────
    existing_pngs = [p for p in slide_pngs if Path(p).exists()]
    missing = len(slide_pngs) - len(existing_pngs)
    if missing:
        logger.warning("carousel_slides_missing", missing=missing, total=len(slide_pngs))

    if not existing_pngs:
        logger.error("carousel_no_slides_rendered")
        return None

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

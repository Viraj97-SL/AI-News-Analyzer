"""
Image generation node — news cards + LinkedIn carousel PDF via html2image.

Produces:
  - 1200x627px individual news cards (used in email newsletter)
  - 1080x1350px portrait carousel slides combined into a PDF (LinkedIn document post)

Carousel slide order:
  1. Cover  — story count + date + topic tags + source count
  2. Story × N — headline + 3 bullet points + image + source names + credibility bar
  3. Closing — CTA + follow prompt
"""

from __future__ import annotations

import base64
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import httpx
from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.agents.nodes.screenshot_utils import capture_slide, make_hti
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"

# Single source of truth for how many stories appear in the carousel AND the LinkedIn post.
# Changing this one constant keeps both in sync.
CAROUSEL_STORY_COUNT = 7

# Browser-style UA required by some hosts (notably Wikimedia, which 403s
# bare/generic UAs per its bot-etiquette policy) — reused for every outbound
# image/page fetch so behavior is consistent and no host silently blocks us.
_HTTP_USER_AGENT = "Mozilla/5.0 (compatible; ai-news-summarizer/1.0; +https://example.com)"

# Color palette updated to match new editorial theme
CATEGORY_COLORS: dict[str, str] = {
    "LLM":             "#3b82f6",
    "Computer Vision": "#f59e0b",
    "Robotics":        "#f43f5e",
    "AI Policy":       "#f97316",
    "AI Startup":      "#10b981",
    "Research":        "#8b5cf6",
    "Industry":        "#06b6d4",
    "Other":           "#6b7280",
}

# Domain → short display name shown in carousel source badge
DOMAIN_DISPLAY_NAMES: dict[str, str] = {
    # Tier 1 — major newspapers / wire
    "nytimes.com":          "NYT",
    "washingtonpost.com":   "WashPost",
    "wsj.com":              "WSJ",
    "ft.com":               "FT",
    "economist.com":        "Economist",
    "bloomberg.com":        "Bloomberg",
    "reuters.com":          "Reuters",
    "apnews.com":           "AP",
    "bbc.com":              "BBC",
    "bbc.co.uk":            "BBC",
    "theguardian.com":      "Guardian",
    # Tier 2 — tech journalism
    "techcrunch.com":       "TechCrunch",
    "venturebeat.com":      "VentureBeat",
    "theverge.com":         "The Verge",
    "wired.com":            "Wired",
    "arstechnica.com":      "Ars Technica",
    "technologyreview.com": "MIT Tech Review",
    "zdnet.com":            "ZDNet",
    "cnet.com":             "CNET",
    "engadget.com":         "Engadget",
    "ieee.org":             "IEEE",
    "thenewstack.io":       "The New Stack",
    "infoq.com":            "InfoQ",
    # Tier 3 — AI labs / companies
    "openai.com":           "OpenAI",
    "anthropic.com":        "Anthropic",
    "deepmind.google":      "DeepMind",
    "deepmind.com":         "DeepMind",
    "blog.google":          "Google",
    "ai.googleblog.com":    "Google AI",
    "huggingface.co":       "HuggingFace",
    "mistral.ai":           "Mistral",
    "meta.com":             "Meta",
    "ai.meta.com":          "Meta AI",
    "blogs.microsoft.com":  "Microsoft",
    "blogs.nvidia.com":     "NVIDIA",
    # Tier 4 — research
    "arxiv.org":            "arXiv",
    "nature.com":           "Nature",
    "science.org":          "Science",
    "openreview.net":       "OpenReview",
    # Tier 5 — newsletters / blogs
    "towardsdatascience.com": "TDS",
    "deeplearning.ai":      "DeepLearning.AI",
    "simonwillison.net":    "Simon Willison",
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
    Split a summary body into up to 3 bullet points, each ending at a complete
    sentence boundary. Never cuts mid-sentence or mid-word.
    """
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    points: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        if len(s) <= 190:
            points.append(s)
        else:
            # Try to end at a sentence-end within the first 190 chars
            chunk = s[:190]
            cut = -1
            for punct in [". ", "! ", "? "]:
                idx = chunk.rfind(punct)
                if idx > 60:
                    cut = idx + 1  # include the punctuation
                    break
            if cut == -1:
                # Fall back to last comma or semicolon
                for punct in [", ", "; "]:
                    idx = chunk.rfind(punct)
                    if idx > 60:
                        cut = idx
                        break
            if cut == -1:
                cut = chunk.rfind(" ")
            if cut > 60:
                points.append(s[:cut].rstrip(" .,") + "…")
            else:
                points.append(chunk.rstrip() + "…")
        if len(points) == 3:
            break
    return points


def _credibility_tier(score: float, source_count: int) -> dict:
    """Map credibility score to a display-safe trust badge."""
    if score >= 0.75:
        return {"label": "Major Outlets", "color": "#10b981", "icon": "◉"}
    elif score >= 0.55:
        return {"label": "Press Coverage", "color": "#f59e0b", "icon": "◎"}
    else:
        return {"label": "Emerging Story", "color": "#6b7280", "icon": "○"}


def _outlet_names_from_urls(source_urls: list[str]) -> list[str]:
    """
    Extract up to 3 recognisable publication names from article URLs.
    Falls back to the bare domain if no display name is registered.
    """
    names: list[str] = []
    seen: set[str] = set()
    for url in source_urls:
        if not url:
            continue
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            continue
        # Exact match first, then parent domain
        name = DOMAIN_DISPLAY_NAMES.get(domain)
        if not name:
            parts = domain.split(".")
            if len(parts) > 2:
                name = DOMAIN_DISPLAY_NAMES.get(".".join(parts[-2:]))
        if not name:
            # Use cleaned domain as fallback (e.g. "techcrunch" from techcrunch.com)
            name = domain.split(".")[0].title() if domain else None
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) == 3:
            break
    return names


def _bias_distribution(source_urls: list[str]) -> dict:
    """
    Compute the political lean distribution of a story's sources using DOMAIN_BIAS.
    Returns counts for left / center / right / unknown and a summary label.
    Based on AllSides / Ad Fontes media bias ratings.
    """
    from app.agents.nodes.credibility import DOMAIN_BIAS

    counts: dict[str, int] = {"left": 0, "center-left": 0, "center": 0, "center-right": 0, "right": 0, "unknown": 0}
    for url in source_urls:
        if not url:
            continue
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            continue
        bias = DOMAIN_BIAS.get(domain)
        if not bias:
            parts = domain.split(".")
            if len(parts) > 2:
                bias = DOMAIN_BIAS.get(".".join(parts[-2:]))
        counts[bias or "unknown"] = counts.get(bias or "unknown", 0) + 1

    total = sum(counts.values()) or 1
    left_n  = counts["left"] + counts["center-left"]
    center_n = counts["center"]
    right_n  = counts["center-right"] + counts["right"]
    unknown_n = counts["unknown"]

    # Summary label
    known = left_n + center_n + right_n
    if known == 0:
        summary = "Independent sources"
    elif center_n / total >= 0.6:
        summary = "Center-dominant"
    elif left_n > right_n and left_n / total >= 0.4:
        summary = "Left-leaning mix"
    elif right_n > left_n and right_n / total >= 0.4:
        summary = "Right-leaning mix"
    else:
        summary = "Balanced coverage"

    return {
        "left": left_n,
        "center": center_n,
        "right": right_n,
        "unknown": unknown_n,
        "total": total,
        # Percentages for the CSS flex bar widths (min 8% so segments are visible)
        "left_pct":    max(8, round(left_n / total * 100))   if left_n   else 0,
        "center_pct":  max(8, round(center_n / total * 100)) if center_n else 0,
        "right_pct":   max(8, round(right_n / total * 100))  if right_n  else 0,
        "summary": summary,
    }


def _reliability_label(pct: int) -> str:
    if pct >= 80:
        return "High credibility"
    if pct >= 60:
        return "Good credibility"
    if pct >= 45:
        return "Mixed sources"
    return "Emerging story"


def _download_image_data_uri(image_url: str) -> str | None:
    """Download an image URL and return a data URI (mime + base64). Returns None on failure."""
    try:
        with httpx.Client(timeout=8, follow_redirects=True) as client:
            resp = client.get(image_url, headers={"User-Agent": _HTTP_USER_AGENT})
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "").split(";")[0].strip()
            if not mime.startswith("image/"):
                return None
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.debug("image_download_failed", url=image_url, error=str(e))
        return None


# Filename fragments that signal a non-editorial image (site chrome, not a
# photo worth showing) — checked case-insensitively against candidate URLs.
_IMG_SKIP_PATTERN = re.compile(
    r"(logo|icon|favicon|avatar|sprite|pixel|spacer|placeholder|badge|1x1|blank\.gif)",
    re.IGNORECASE,
)

# Meta/link tags that carry a representative article image, in priority order.
# Each entry is (attribute-match regex, attribute-then-content regex) so we
# catch both `<meta property=X content=Y>` and `<meta content=Y property=X>`
# tag orderings, which vary across CMSs.
_META_IMAGE_PATTERNS: list[tuple[str, str]] = [
    (
        r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']',
    ),
    (
        r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image(?::src)?["\']',
    ),
    (
        r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
        r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']image_src["\']',
    ),
]


def _fetch_page_html(url: str) -> str | None:
    """Fetch a page's raw HTML. Returns None on any network failure."""
    try:
        with httpx.Client(timeout=6, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": _HTTP_USER_AGENT})
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.debug("page_fetch_failed", url=url, error=str(e))
        return None


def _extract_meta_image_url(html: str) -> str | None:
    """Scan og:image / twitter:image / link[rel=image_src] tags, first match wins."""
    for attr_first, content_first in _META_IMAGE_PATTERNS:
        m = re.search(attr_first, html, re.IGNORECASE) or re.search(content_first, html, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if candidate and not _IMG_SKIP_PATTERN.search(candidate):
                return candidate
    return None


def _extract_body_image_url(html: str) -> str | None:
    """
    Fallback: scan article-body <img> tags for the first substantial, real
    photo, skipping obvious site chrome (logos, icons, tracking pixels).
    Used when a page exposes no usable og:image/twitter:image meta tags.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None

    for img in soup.find_all("img"):
        src = (img.get("src") or img.get("data-src") or img.get("data-lazy-src") or "").strip()
        if not src or src.startswith("data:") or _IMG_SKIP_PATTERN.search(src):
            continue

        for dim_attr in ("width", "height"):
            dim = img.get(dim_attr)
            if dim:
                try:
                    if int(re.sub(r"[^\d]", "", dim) or 0) < 150:
                        src = ""
                        break
                except ValueError:
                    pass
        if not src:
            continue

        return src
    return None


def _fetch_og_image(url: str) -> str | None:
    """
    Fetch an editorial photo for an article URL and return a data URI.

    Tries, in order, against a single page fetch: og:image, twitter:image,
    link[rel=image_src], then the first substantial <img> in the body.
    Returns None only if the page is unreachable or no candidate downloads.
    """
    html = _fetch_page_html(url)
    if not html:
        return None

    candidates: list[tuple[str, str]] = []
    meta_url = _extract_meta_image_url(html)
    if meta_url:
        candidates.append(("meta_tag", urljoin(url, meta_url)))
    body_url = _extract_body_image_url(html)
    if body_url:
        candidates.append(("body_img", urljoin(url, body_url)))

    for method, image_url in candidates:
        if not image_url.startswith("http"):
            continue
        data_uri = _download_image_data_uri(image_url)
        if data_uri:
            logger.debug("story_image_method_used", method=method, url=url)
            return data_uri

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


def _wikipedia_image_search(query: str) -> str | None:
    """
    Keyless fallback: pull the lead image from the best-matching Wikipedia
    article via the public MediaWiki API. No API key required, so unlike
    Serper this always works in production regardless of configuration —
    a topical stock photo beats an empty gradient when nothing else lands.
    """
    try:
        with httpx.Client(timeout=6) as client:
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": query,
                    "gsrlimit": 1,
                    "prop": "pageimages",
                    "piprop": "original",
                    "format": "json",
                },
                headers={"User-Agent": _HTTP_USER_AGENT},
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})

        for page in pages.values():
            image_url = page.get("original", {}).get("source", "")
            if image_url and image_url.startswith("http"):
                data_uri = _download_image_data_uri(image_url)
                if data_uri:
                    logger.debug("wikipedia_image_found", query=query[:50])
                    return data_uri

    except Exception as e:
        logger.debug("wikipedia_image_search_failed", query=query[:50], error=str(e))
    return None


def _fetch_story_image(source_url: str, headline: str) -> str | None:
    """
    Get a relevant image for a story slide, trying progressively looser
    methods until one produces a real downloadable photo:

    1. The article's own page: og:image → twitter:image → link[image_src]
       → first substantial <img> in the body (all from one page fetch).
    2. Serper Google Images search for the headline (needs SERPER_API_KEY;
       silently skipped if not configured).
    3. Wikipedia's lead image for the best-matching article (keyless,
       always available) — a relevant stock photo beats no photo.
    4. Returns None → template renders the category gradient fallback.
    """
    if source_url:
        data_uri = _fetch_og_image(source_url)
        if data_uri:
            return data_uri

    data_uri = _serper_image_search(headline)
    if data_uri:
        return data_uri

    return _wikipedia_image_search(headline)


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
    return make_hti(OUTPUT_DIR, size)


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
        path = capture_slide(hti, html, filename, label=f"card_{i}", output_dir=OUTPUT_DIR)
        if path:
            paths.append(path)

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# LinkedIn carousel (1080 × 1080 slides → PDF)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_carousel_pdf(summaries: list[dict], run_id: str, env: Environment) -> tuple[str, list[str]] | None:
    """
    Render infographic carousel slides (portrait 1080×1350) and combine into a PDF.

    Slide order:
      1  Cover     — story count + date + author branding
      2… Stories   — headline, 3 bullets, real article photo, source names
      N  Closing   — CTA + follow prompt
    """
    template = env.get_template("carousel_slide.html")
    hti = _make_hti((1080, 1350))  # portrait 4:5 — better for LinkedIn mobile

    story_summaries = summaries[:CAROUSEL_STORY_COUNT]
    total_slides = len(story_summaries)
    date_str = date.today().strftime("%B %d, %Y")
    slide_pngs: list[str] = []

    # Pre-compute cover-slide extras from all summaries
    cover_topics: list[str] = sorted({
        s.get("category", "")
        for s in story_summaries
        if s.get("category") and s.get("category") not in ("Other", "")
    })[:5]
    if not cover_topics:
        cover_topics = ["AI", "Technology"]

    all_source_urls = [url for s in story_summaries for url in s.get("source_urls", []) if url]
    total_source_count = len({
        urlparse(u).netloc.lower().replace("www.", "")
        for u in all_source_urls
        if u
    })

    # ── 1. Cover slide ────────────────────────────────────────
    html = template.render(
        slide_type="cover",
        story_count=total_slides,
        date_str=date_str,
        cover_topics=cover_topics,
        total_source_count=total_source_count,
    )
    name = f"carousel_{run_id}_cover.png"
    path = capture_slide(hti, html, name, label="cover", output_dir=OUTPUT_DIR)
    if path:
        slide_pngs.append(path)

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

        outlet_names = _outlet_names_from_urls(source_urls)
        bias_dist = _bias_distribution(source_urls)
        reliability_pct = int(cred_score * 100)
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
            outlet_names=outlet_names,
            bias_dist=bias_dist,
            reliability_pct=reliability_pct,
            reliability_label=_reliability_label(reliability_pct),
        )
        name = f"carousel_{run_id}_{i}.png"
        path = capture_slide(hti, html, name, label=f"story_{i}", output_dir=OUTPUT_DIR)
        if path:
            slide_pngs.append(path)

    # ── 3. Closing / CTA slide ────────────────────────────────
    html = template.render(slide_type="closing")
    name = f"carousel_{run_id}_close.png"
    path = capture_slide(hti, html, name, label="closing", output_dir=OUTPUT_DIR)
    if path:
        slide_pngs.append(path)

    # ── Combine PNGs → PDF ────────────────────────────────────
    total_planned = total_slides + 2  # cover + stories + closing
    existing_pngs = slide_pngs
    missing = total_planned - len(existing_pngs)
    if missing:
        logger.warning("carousel_slides_missing", missing=missing, total=total_planned)

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

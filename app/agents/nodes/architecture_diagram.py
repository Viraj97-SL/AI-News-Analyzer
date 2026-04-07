"""
Architecture diagram node — multi-strategy figure extraction from ArXiv papers.

Extraction strategies (tried in order):
  1. ArXiv HTML version  — labeled <figure> elements with captions; best quality.
  2. PyMuPDF PDF         — improved heuristics: all pages, size + keyword scoring.
  3. LLM ASCII fallback  — generated box diagram when no images can be fetched.

All extracted figures are stored in state['paper_figures'] as
  [{"b64": "<base64-png>", "caption": "<fig caption text>"}]

The first (best) figure is also stored in the legacy
  architecture_diagram_b64 / architecture_diagram_path fields
for backwards compatibility with the research card template.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_arxiv_id(paper_url: str) -> str:
    """Return bare ArXiv ID (e.g. '2301.07685') from any URL format."""
    raw = paper_url.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", raw)


# ── Strategy 1: ArXiv HTML figures ───────────────────────────────────────────

def _fetch_html_figures(arxiv_id: str) -> list[tuple[bytes, str]]:
    """
    Download labeled figures from the ArXiv HTML version of a paper.
    Returns list of (image_bytes, caption_text); empty list on failure.

    ArXiv started generating HTML5 exports (ar5iv / arxiv HTML) in 2023.
    The images are served as relative paths under
    https://arxiv.org/html/{id}/extracted_images/...
    """
    import httpx

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    base_url = f"https://arxiv.org/html/{arxiv_id}/"

    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(html_url, headers={"User-Agent": "research-analyzer/1.0"})
            if resp.status_code != 200:
                logger.debug("arxiv_html_unavailable", arxiv_id=arxiv_id, status=resp.status_code)
                return []
            html_text = resp.text
    except Exception as e:
        logger.debug("arxiv_html_fetch_error", arxiv_id=arxiv_id, error=str(e))
        return []

    # Lightweight regex parsing — avoids BeautifulSoup dependency
    figure_re = re.compile(r"<figure[^>]*>(.*?)</figure>", re.DOTALL | re.IGNORECASE)
    img_re = re.compile(r'<img[^>]*\ssrc=["\']([^"\']+)["\']', re.IGNORECASE)
    caption_re = re.compile(r"<figcaption[^>]*>(.*?)</figcaption>", re.DOTALL | re.IGNORECASE)
    tag_re = re.compile(r"<[^>]+>")

    results: list[tuple[bytes, str]] = []

    with httpx.Client(timeout=15, follow_redirects=True) as client:
        for fig_m in figure_re.finditer(html_text):
            fig_html = fig_m.group(1)

            img_m = img_re.search(fig_html)
            if not img_m:
                continue

            src = img_m.group(1)
            if src.startswith("data:"):
                continue  # skip inline data URIs (usually tiny icons)

            if not src.startswith("http"):
                src = base_url + src.lstrip("./")

            cap_m = caption_re.search(fig_html)
            caption = (
                tag_re.sub("", cap_m.group(1)).strip()[:250] if cap_m else ""
            )

            try:
                img_resp = client.get(src, headers={"User-Agent": "research-analyzer/1.0"})
                if img_resp.status_code == 200 and len(img_resp.content) > 500:
                    results.append((img_resp.content, caption))
                    if len(results) >= 4:
                        break
            except Exception:
                continue

    if results:
        logger.info("html_figures_extracted", arxiv_id=arxiv_id, count=len(results))
    return results


# ── Strategy 2: PyMuPDF PDF extraction ───────────────────────────────────────

def _fetch_pdf_figures(paper_url: str) -> list[tuple[bytes, str]]:
    """
    Extract figures from the ArXiv PDF using PyMuPDF.

    Improvements over the original implementation:
      - Scans up to 10 pages (was 5).
      - Scores each image by area + keyword context + page position.
      - Rejects extreme aspect ratios (headers/footers/logos).
      - Returns up to 3 best figures (was 1).
    """
    try:
        import fitz  # type: ignore[import]  # PyMuPDF
        import httpx
        from PIL import Image as PILImage

        pdf_url = paper_url.replace("/abs/", "/pdf/")
        logger.info("fetching_arxiv_pdf", url=pdf_url)

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(pdf_url, headers={"User-Agent": "research-analyzer/1.0"})
            resp.raise_for_status()

        doc = fitz.Document(stream=resp.content, filetype="pdf")  # type: ignore[call-arg]
        candidates: list[tuple[float, bytes]] = []  # (score, raw_bytes)

        diagram_keywords = frozenset(
            ["figure", "fig.", "architecture", "overview", "framework",
             "pipeline", "model", "diagram", "workflow", "approach"]
        )

        for page_idx in range(min(10, len(doc))):
            page = doc[page_idx]
            page_text = page.get_text().lower()
            has_diagram_context = any(kw in page_text for kw in diagram_keywords)

            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_data = doc.extract_image(xref)
                w, h = img_data.get("width", 0), img_data.get("height", 0)

                # Reject tiny images and extreme aspect ratios
                if w < 200 or h < 150:
                    continue
                aspect = w / h if h > 0 else 0
                if aspect > 5.0 or aspect < 0.15:
                    continue  # skip banners and thin strips

                # Score: pixel area + diagram context bonus + early-page bonus
                score = (
                    float(w * h)
                    + (8_000.0 if has_diagram_context else 0.0)
                    + max(0.0, (6 - page_idx) * 2_000.0)
                )
                candidates.append((score, img_data["image"]))

        doc.close()
        candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[tuple[bytes, str]] = []
        for _, raw_bytes in candidates[:3]:
            try:
                pil = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
                pil.thumbnail((800, 600), PILImage.LANCZOS)  # type: ignore[attr-defined]
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                results.append((buf.getvalue(), ""))
            except Exception:
                continue

        if results:
            logger.info("pdf_figures_extracted", count=len(results))
        return results

    except ImportError:
        logger.info("pymupdf_not_installed", hint="pip install pymupdf")
        return []
    except Exception as e:
        logger.warning("pdf_figure_extraction_failed", error=str(e))
        return []


# ── Strategy 3: LLM ASCII fallback ───────────────────────────────────────────

def _ascii_fallback(methodology: str) -> str:
    """Generate an ASCII architecture diagram via Gemini Flash; return styled HTML."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=settings.google_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a compact ASCII box diagram showing the data / processing flow described. "
            "Use → arrows between boxes. Max 8 boxes. Return ONLY the diagram text, no explanation.",
        ),
        ("user", "{methodology}"),
    ])
    _resp = (prompt | llm).invoke({"methodology": methodology[:800]}).content
    text = (
        "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in _resp).strip()
        if isinstance(_resp, list) else _resp.strip()
    )
    return (
        '<pre style="color:#00f3ff;font-family:\'JetBrains Mono\',monospace;'
        "font-size:11px;background:rgba(0,243,255,0.04);padding:14px;"
        "border-radius:6px;border:1px solid rgba(0,243,255,0.15);"
        f'overflow:hidden;line-height:1.6;margin:0;white-space:pre-wrap">{text}</pre>'
    )


# ── Main node ─────────────────────────────────────────────────────────────────

def architecture_diagram_node(state: "PipelineState") -> dict:
    """
    Extract figures from the chosen ArXiv paper.

    Returns:
      architecture_diagram_b64   — primary figure (base64 PNG) or "".
      architecture_diagram_path  — saved PNG path or "".
      architecture_fallback_text — ASCII HTML if no images found, else "".
      paper_figures              — all extracted figures as [{b64, caption}].
    """
    paper = state.get("chosen_research_paper", {})
    analysis = state.get("deep_analysis", {})
    run_id = state.get("run_id", "dev")

    paper_url = paper.get("url", "")
    arxiv_id = _extract_arxiv_id(paper_url) if paper_url else ""
    methodology = analysis.get("methodology", "")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    empty_result = {
        "architecture_diagram_path": "",
        "architecture_diagram_b64": "",
        "architecture_fallback_text": "",
        "paper_figures": [],
    }

    # ── Strategy 1: ArXiv HTML ─────────────────────────────────────────
    figures: list[tuple[bytes, str]] = []
    if arxiv_id:
        figures = _fetch_html_figures(arxiv_id)

    # ── Strategy 2: PyMuPDF PDF ────────────────────────────────────────
    if not figures and paper_url:
        figures = _fetch_pdf_figures(paper_url)

    # ── Successful extraction ──────────────────────────────────────────
    if figures:
        primary_bytes, primary_caption = figures[0]

        # Save primary figure to disk (for email attachments)
        primary_path = str(OUTPUT_DIR / f"arch_diagram_{run_id}.png")
        Path(primary_path).write_bytes(primary_bytes)
        primary_b64 = base64.b64encode(primary_bytes).decode()

        paper_figures = [
            {"b64": base64.b64encode(img_b).decode(), "caption": cap}
            for img_b, cap in figures
        ]

        logger.info(
            "architecture_figures_ready",
            count=len(figures),
            has_captions=any(f["caption"] for f in paper_figures),
        )
        return {
            "architecture_diagram_path": primary_path,
            "architecture_diagram_b64": primary_b64,
            "architecture_fallback_text": "",
            "paper_figures": paper_figures,
            "current_step": "architecture_diagram_extracted",
        }

    # ── Strategy 3: ASCII fallback ─────────────────────────────────────
    logger.info("no_figures_found_using_ascii_fallback")
    try:
        fallback_html = _ascii_fallback(methodology)
        return {
            **empty_result,
            "architecture_fallback_text": fallback_html,
            "current_step": "architecture_fallback_generated",
        }
    except Exception as e:
        logger.error("ascii_fallback_failed", error=str(e))
        return {**empty_result, "current_step": "architecture_diagram_skipped"}

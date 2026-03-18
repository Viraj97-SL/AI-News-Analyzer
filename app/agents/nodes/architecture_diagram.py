"""
Architecture diagram node — fetches the first figure from the ArXiv PDF via PyMuPDF,
or falls back to an LLM-generated ASCII box diagram.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.state import PipelineState

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OUTPUT_DIR = Path("./output/images")


def _abs_to_pdf_url(abs_url: str) -> str:
    """Convert https://arxiv.org/abs/XXXX → https://arxiv.org/pdf/XXXX."""
    return abs_url.replace("/abs/", "/pdf/")


def _ascii_fallback(methodology: str) -> str:
    """Call Gemini Flash to produce an ASCII box-diagram; return styled HTML."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=settings.google_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Generate a compact ASCII box diagram showing the data / processing flow described. "
         "Use → arrows between boxes. Max 8 boxes. Return ONLY the diagram text, no explanation."),
        ("user", "{methodology}"),
    ])
    result = (prompt | llm).invoke({"methodology": methodology[:800]})
    text = result.content.strip()
    return (
        f'<pre style="color:#00f3ff;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:11px;background:rgba(0,243,255,0.04);padding:14px;'
        f'border-radius:6px;border:1px solid rgba(0,243,255,0.15);'
        f'overflow:hidden;line-height:1.6;margin:0;white-space:pre-wrap">{text}</pre>'
    )


def architecture_diagram_node(state: "PipelineState") -> dict:
    """Try PyMuPDF figure extraction; fall back to LLM ASCII diagram."""
    paper = state.get("chosen_research_paper", {})
    analysis = state.get("deep_analysis", {})
    run_id = state.get("run_id", "dev")

    paper_url = paper.get("url", "")
    methodology = analysis.get("methodology", "")

    empty = {
        "architecture_diagram_path": "",
        "architecture_diagram_b64": "",
        "architecture_fallback_text": "",
    }

    # ── Primary: PyMuPDF figure extraction ───────────────────────────────
    try:
        import fitz  # type: ignore[import]  # PyMuPDF
        import httpx
        from PIL import Image as PILImage

        pdf_url = _abs_to_pdf_url(paper_url)
        logger.info("fetching_arxiv_pdf", url=pdf_url)

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(pdf_url, headers={"User-Agent": "research-analyzer/1.0"})
            resp.raise_for_status()
            pdf_bytes = resp.content

        doc = fitz.Document(stream=pdf_bytes, filetype="pdf")  # type: ignore[call-arg]

        found_img_bytes: bytes | None = None
        for page_idx in range(min(5, len(doc))):
            page = doc[page_idx]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_data = doc.extract_image(xref)
                w, h = img_data.get("width", 0), img_data.get("height", 0)
                if w >= 200 and h >= 150:
                    found_img_bytes = img_data["image"]
                    break
            if found_img_bytes:
                break
        doc.close()

        if found_img_bytes:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pil_img = PILImage.open(io.BytesIO(found_img_bytes)).convert("RGB")

            # Resize to max 800×480 preserving aspect ratio
            pil_img.thumbnail((800, 480), PILImage.LANCZOS)  # type: ignore[attr-defined]

            diagram_path = str(OUTPUT_DIR / f"arch_diagram_{run_id}.png")
            pil_img.save(diagram_path)

            b64 = base64.b64encode(Path(diagram_path).read_bytes()).decode()
            logger.info("architecture_diagram_extracted", path=diagram_path)
            return {
                "architecture_diagram_path": diagram_path,
                "architecture_diagram_b64": b64,
                "architecture_fallback_text": "",
                "current_step": "architecture_diagram_extracted",
            }

        logger.info("no_suitable_figure_in_pdf")

    except ImportError:
        logger.info("pymupdf_not_installed", hint="pip install pymupdf — using ASCII fallback")
    except Exception as e:
        logger.warning("pdf_figure_extraction_failed", error=str(e))

    # ── Fallback: ASCII diagram via LLM ──────────────────────────────────
    try:
        fallback_html = _ascii_fallback(methodology)
        logger.info("ascii_diagram_generated")
        return {
            **empty,
            "architecture_fallback_text": fallback_html,
            "current_step": "architecture_fallback_generated",
        }
    except Exception as e:
        logger.error("ascii_fallback_failed", error=str(e))
        return {**empty, "current_step": "architecture_diagram_skipped"}

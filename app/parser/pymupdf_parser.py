"""PyMuPDF digital PDF parser (P1)."""
from __future__ import annotations

import fitz  # PyMuPDF

from app.cleaner.unicode import clean_text, indic_script_ratio
from app.core.logging import get_logger
from app.pipeline.types import ParsedPage, TextBlock

log = get_logger(__name__)


def _blocks_from_page(page: fitz.Page, page_number: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    try:
        d = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return blocks
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines = b.get("lines", [])
        parts: list[str] = []
        max_size = 0.0
        for line in lines:
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    parts.append(t)
                    max_size = max(max_size, float(span.get("size", 0)))
        text = " ".join(parts).strip()
        if not text:
            continue
        text, _ = clean_text(text)
        btype = "paragraph"
        level = None
        if max_size >= 14 and len(text.split()) <= 15:
            btype = "heading"
            if max_size >= 18:
                level = 1
            elif max_size >= 15:
                level = 2
            else:
                level = 3
        blocks.append(
            TextBlock(
                block_type=btype,
                text=text,
                page_start=page_number,
                page_end=page_number,
                font_size=max_size,
                heading_level=level,
            )
        )
    return blocks


def parse_digital_pdf(pdf_bytes: bytes) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i in range(len(doc)):
            page = doc[i]
            pn = i + 1
            text = page.get_text("text") or ""
            text, _ = clean_text(text, preserve_line_breaks=True)
            blocks = _blocks_from_page(page, pn)
            ratio = indic_script_ratio(text)
            pages.append(
                ParsedPage(
                    page_number=pn,
                    text=text,
                    confidence=0.98 if ratio >= 0.15 else 0.5,
                    engine="pymupdf",
                    blocks=blocks,
                    indic_ratio=ratio,
                )
            )
    finally:
        doc.close()
    log.info("parser.pymupdf", pages=len(pages))
    return pages


def extract_bookmarks(pdf_bytes: bytes) -> list[tuple[int, str, int]]:
    """Return (level, title, page_number) from PDF TOC."""
    out: list[tuple[int, str, int]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        toc = doc.get_toc(simple=True) or []
        for level, title, page in toc:
            out.append((level, str(title).strip(), int(page)))
    finally:
        doc.close()
    return out

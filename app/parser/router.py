"""Document type detection and parser routing (P1)."""
from __future__ import annotations

from app.core.logging import get_logger
from app.parser.pymupdf_parser import extract_bookmarks, parse_digital_pdf
from app.parser.tables import extract_tables
from app.pipeline.types import DocumentProfile, ParsedPage, TextBlock
from app.cleaner.unicode import indic_script_ratio

log = get_logger(__name__)


def detect_profile(pdf_bytes: bytes) -> DocumentProfile:
    pages = parse_digital_pdf(pdf_bytes)
    if not pages:
        return DocumentProfile(document_type="scanned_pdf", page_count=0)
    ratios = [p.indic_ratio for p in pages if p.text.strip()]
    avg = sum(ratios) / len(ratios) if ratios else 0.0
    digital = sum(
        1 for p in pages if p.engine == "pymupdf" and len(p.text.strip()) >= 20
    )
    scanned = len(pages) - digital
    bookmarks = bool(extract_bookmarks(pdf_bytes))
    tables = extract_tables(pdf_bytes)
    has_tables = bool(tables)
    if digital == len(pages):
        dtype = "digital_pdf"
    elif scanned == len(pages):
        dtype = "scanned_pdf"
    else:
        dtype = "mixed"
    if has_tables and dtype == "digital_pdf":
        dtype = "table_heavy"
    return DocumentProfile(
        document_type=dtype,
        has_bookmarks=bookmarks,
        has_tables=has_tables,
        avg_indic_ratio=avg,
        page_count=len(pages),
    )


def _merge_table_blocks(pages: list[ParsedPage], tables: dict[int, list[TextBlock]]) -> list[ParsedPage]:
    out: list[ParsedPage] = []
    for page in pages:
        blocks = list(page.blocks)
        blocks.extend(tables.get(page.page_number, []))
        if tables.get(page.page_number):
            table_text = "\n\n".join(b.text for b in tables[page.page_number])
            text = (page.text + "\n\n" + table_text).strip() if page.text else table_text
        else:
            text = page.text
        out.append(
            ParsedPage(
                page_number=page.page_number,
                text=text,
                confidence=page.confidence,
                engine=page.engine,
                blocks=blocks,
                indic_ratio=indic_script_ratio(text),
            )
        )
    return out


def parse_document(pdf_bytes: bytes, profile: DocumentProfile | None = None) -> tuple[DocumentProfile, list[ParsedPage]]:
    profile = profile or detect_profile(pdf_bytes)
    pages = parse_digital_pdf(pdf_bytes)

    # Scanned / weak pages: delegate to OCR pipeline (filled by ingest after OCR).
    if profile.document_type in ("scanned_pdf", "mixed") and profile.avg_indic_ratio < 0.15:
        log.info("parser.route_ocr", document_type=profile.document_type)
        return profile, pages

    if profile.has_tables:
        tables = extract_tables(pdf_bytes)
        pages = _merge_table_blocks(pages, tables)

    log.info(
        "parser.done",
        document_type=profile.document_type,
        pages=len(pages),
        avg_indic=round(profile.avg_indic_ratio, 3),
    )
    return profile, pages


def pages_from_ocr(ocr_pages: list[tuple[int, str, float]], engine: str = "ocr") -> list[ParsedPage]:
    """Convert legacy OCR tuples to ParsedPage list."""
    out: list[ParsedPage] = []
    for pn, text, conf in ocr_pages:
        ratio = indic_script_ratio(text)
        blocks = [
            TextBlock(
                block_type="paragraph",
                text=ln,
                page_start=pn,
                page_end=pn,
                confidence=conf,
            )
            for ln in text.split("\n")
            if ln.strip()
        ]
        out.append(
            ParsedPage(
                page_number=pn,
                text=text,
                confidence=conf,
                engine=engine,
                blocks=blocks,
                indic_ratio=ratio,
            )
        )
    return out

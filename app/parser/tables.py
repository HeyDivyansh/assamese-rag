"""Table extraction via pdfplumber (P4)."""
from __future__ import annotations

import io

import pdfplumber

from app.core.logging import get_logger
from app.pipeline.types import TextBlock

log = get_logger(__name__)


def _table_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        cells = [str(c or "").strip().replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_tables(pdf_bytes: bytes) -> dict[int, list[TextBlock]]:
    """page_number -> table blocks."""
    by_page: dict[int, list[TextBlock]] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception:  # noqa: BLE001
                tables = []
            blocks: list[TextBlock] = []
            for table in tables:
                md = _table_to_markdown(table)
                if len(md.strip()) < 10:
                    continue
                blocks.append(
                    TextBlock(
                        block_type="table",
                        text=md,
                        page_start=i,
                        page_end=i,
                        confidence=0.9,
                    )
                )
            if blocks:
                by_page[i] = blocks
    if by_page:
        log.info("parser.tables", pages=len(by_page))
    return by_page

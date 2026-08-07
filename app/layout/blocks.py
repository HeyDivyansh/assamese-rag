"""Layout block utilities (P2)."""
from __future__ import annotations

import re

from app.pipeline.types import ParsedPage, TextBlock

_LIST_ITEM = re.compile(r"^[\s]*(?:[-•*]|\d+[.)]|[(][a-zA-Z০-৯][)]|\([ivx]+\))\s+")
_QA_LINE = re.compile(r".*\?\s*$")


def classify_blocks(pages: list[ParsedPage]) -> list[ParsedPage]:
    """Refine block types: list, qa, paragraph."""
    out: list[ParsedPage] = []
    for page in pages:
        blocks: list[TextBlock] = []
        if page.blocks:
            for b in page.blocks:
                text = b.text.strip()
                btype = b.block_type
                if _LIST_ITEM.match(text):
                    btype = "list"
                elif _QA_LINE.match(text):
                    btype = "qa"
                blocks.append(
                    TextBlock(
                        block_type=btype,
                        text=text,
                        page_start=b.page_start,
                        page_end=b.page_end,
                        confidence=b.confidence,
                        heading_level=b.heading_level,
                        font_size=b.font_size,
                    )
                )
        else:
            for ln in page.text.split("\n"):
                ln = ln.strip()
                if not ln:
                    continue
                btype = "paragraph"
                if _LIST_ITEM.match(ln):
                    btype = "list"
                elif _QA_LINE.match(ln):
                    btype = "qa"
                blocks.append(
                    TextBlock(
                        block_type=btype,
                        text=ln,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        confidence=page.confidence,
                    )
                )
        out.append(
            ParsedPage(
                page_number=page.page_number,
                text=page.text,
                confidence=page.confidence,
                engine=page.engine,
                blocks=blocks,
                indic_ratio=page.indic_ratio,
            )
        )
    return out


def blocks_to_flat_text(blocks: list[TextBlock]) -> str:
    parts: list[str] = []
    for b in blocks:
        if b.block_type == "table":
            parts.append(b.text)
        elif b.block_type == "list":
            parts.append(b.text)
        else:
            parts.append(b.text)
    return "\n\n".join(parts)

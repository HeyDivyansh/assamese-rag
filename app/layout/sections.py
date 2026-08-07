"""Semantic section detection from layout blocks (P2)."""
from __future__ import annotations

import re
import uuid

from app.core.logging import get_logger
from app.layout.blocks import blocks_to_flat_text
from app.pipeline.types import ParsedPage, Section, TextBlock

log = get_logger(__name__)

_HEADING_RE = re.compile(r"^\s*(?:\d+[.)]\s+|[\u0966-\u096F]+[.)]\s+)?\S.{0,100}$")
_SENTENCE_END = ("।", ".", "?", "!", ":", ";")


def _looks_like_heading(text: str, block: TextBlock | None = None) -> bool:
    line = text.strip()
    if not line or len(line) > 120:
        return False
    if block and block.block_type == "heading":
        return True
    if line.endswith(_SENTENCE_END):
        return False
    words = line.split()
    if len(words) > 15:
        return False
    return bool(_HEADING_RE.match(line)) or (block and block.heading_level is not None)


def detect_sections(pages: list[ParsedPage]) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None
    path: list[str] = []

    def _flush():
        nonlocal current
        if current and (current.text.strip() or current.title):
            sections.append(current)
        current = None

    for page in pages:
        blocks = page.blocks or [
            TextBlock("paragraph", page.text, page.page_number, page.page_number)
        ]
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if _looks_like_heading(text, block):
                _flush()
                level = block.heading_level or 1
                path = path[: max(0, level - 1)] + [text]
                current = Section(
                    section_id=uuid.uuid4(),
                    title=text,
                    heading_path=list(path),
                    text="",
                    page_start=block.page_start,
                    page_end=block.page_end,
                    blocks=[block],
                )
            else:
                if current is None:
                    current = Section(
                        section_id=uuid.uuid4(),
                        title=None,
                        heading_path=list(path),
                        text="",
                        page_start=block.page_start,
                        page_end=block.page_end,
                        blocks=[],
                    )
                current.blocks.append(block)
                current.page_end = max(current.page_end, block.page_end)
                sep = "\n\n" if block.block_type == "table" else "\n"
                current.text = (current.text + sep + text).strip() if current.text else text
    _flush()

    if not sections:
        joined = blocks_to_flat_text(
            [b for p in pages for b in (p.blocks or [])]
        ) or "\n\n".join(p.text for p in pages if p.text.strip())
        first = pages[0].page_number if pages else 1
        last = pages[-1].page_number if pages else first
        sections = [
            Section(
                section_id=uuid.uuid4(),
                title=None,
                heading_path=[],
                text=joined,
                page_start=first,
                page_end=last,
            )
        ]
    log.info("layout.sections", count=len(sections))
    return sections

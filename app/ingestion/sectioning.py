"""Section / layout detection.

Primary: PP-StructureV3 (Paddle layout model) to recover headings & blocks.
Fallback: font-size / heading heuristics on the cleaned text.

Both paths yield a flat list of `Section`s carrying a title and the text that
belongs under it, tagged with the page it started on. Chunking later respects
these boundaries.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Section:
    section_id: uuid.UUID
    title: str | None
    text: str
    page_number: int
    blocks: list[str] = field(default_factory=list)


# A heading heuristic: short line, no terminal punctuation, often numbered.
_HEADING_RE = re.compile(r"^\s*(?:\d+[.)]\s+)?\S.{0,80}$")
_SENTENCE_END = ("।", ".", "?", "!", ":", ";")  # includes Devanagari/Bengali danda


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line.endswith(_SENTENCE_END):
        return False
    words = line.split()
    return len(words) <= 12 and bool(_HEADING_RE.match(line))


def _heuristic_sections(pages: list[tuple[int, str, float]]) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None

    def _flush():
        nonlocal current
        if current is not None and (current.text.strip() or current.title):
            sections.append(current)
        current = None

    for page_number, text, _conf in pages:
        for para in text.split("\n\n"):
            lines = [ln for ln in para.split("\n") if ln.strip()]
            if not lines:
                continue
            first = lines[0]
            if _looks_like_heading(first):
                _flush()
                current = Section(
                    section_id=uuid.uuid4(),
                    title=first.strip(),
                    text="\n".join(lines[1:]).strip(),
                    page_number=page_number,
                )
            else:
                if current is None:
                    current = Section(
                        section_id=uuid.uuid4(),
                        title=None,
                        text="",
                        page_number=page_number,
                    )
                current.text = (current.text + "\n\n" + para).strip()
    _flush()

    if not sections:
        # Whole doc as a single untitled section.
        joined = "\n\n".join(t for _, t, _ in pages if t.strip())
        first_page = pages[0][0] if pages else 1
        sections = [Section(uuid.uuid4(), None, joined, first_page)]
    return sections


def _ppstructure_sections(pdf_bytes: bytes) -> list[Section] | None:
    """Attempt PP-StructureV3. Returns None if unavailable/underperforms."""
    try:
        from paddleocr import PPStructure  # type: ignore
    except Exception:  # noqa: BLE001
        log.warning("sectioning.ppstructure_unavailable")
        return None
    # TODO(layout): Full PP-StructureV3 wiring requires rendering pages and
    # mapping layout regions back to text; for Bengali-script docs its accuracy
    # is unverified, so we currently prefer the robust heuristic path and keep
    # this as an explicit extension point.
    return None


def detect_sections(
    pages: list[tuple[int, str, float]], pdf_bytes: bytes | None = None
) -> list[Section]:
    if pdf_bytes is not None:
        via_layout = _ppstructure_sections(pdf_bytes)
        if via_layout:
            log.info("sectioning.used_ppstructure", sections=len(via_layout))
            return via_layout
    sections = _heuristic_sections(pages)
    log.info("sectioning.used_heuristics", sections=len(sections))
    return sections

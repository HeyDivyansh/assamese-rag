"""Adaptive semantic chunking (P0/P2)."""
from __future__ import annotations

import hashlib
import re
import uuid

from app.chunker.dedup import deduplicate_chunks, filter_noise, merge_tiny_chunks
from app.cleaner.unicode import clean_text, detect_language, is_embeddable, quality_score
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.embedding import count_tokens
from app.pipeline.types import ChunkDraft, Section, TextBlock

log = get_logger(__name__)

_SPLIT_RE = re.compile(r"(?<=[।\.\?\!])\s+")
_LIST_ITEM = re.compile(r"^[\s]*(?:[-•*]|\d+[.)])\s+")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_qa_units(lines: list[str]) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("?") or line.endswith("？"):
            ans = ""
            if i + 1 < len(lines) and not lines[i + 1].endswith("?"):
                ans = lines[i + 1]
                i += 2
            else:
                i += 1
            units.append(("qa", f"{line}\n{ans}".strip()))
        else:
            units.append(("paragraph", line))
            i += 1
    return units


def _units_from_section(section: Section) -> list[tuple[str, str, TextBlock | None]]:
    units: list[tuple[str, str, TextBlock | None]] = []
    if section.blocks:
        buf: list[str] = []
        btype = "paragraph"
        block: TextBlock | None = None
        for b in section.blocks:
            if b.block_type == "table":
                if buf:
                    units.append((btype, "\n".join(buf), block))
                    buf = []
                units.append(("table", b.text, b))
                continue
            if b.block_type == "list":
                if buf and btype != "list":
                    units.append((btype, "\n".join(buf), block))
                    buf = []
                btype = "list"
                buf.append(b.text)
                block = b
                continue
            if b.block_type == "qa":
                if buf:
                    units.append((btype, "\n".join(buf), block))
                    buf = []
                units.append(("qa", b.text, b))
                btype = "paragraph"
                block = None
                continue
            if b.block_type == "heading":
                continue
            buf.append(b.text)
            block = b
        if buf:
            units.append((btype, "\n".join(buf), block))
        return units

    body = section.text or ""
    if section.title:
        body = f"{section.title}\n{body}".strip()
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if sum(1 for ln in lines if ln.endswith("?")) >= 2:
        for utype, utext in _split_qa_units(lines):
            units.append((utype, utext, None))
        return units
    for para in re.split(r"\n{2,}", body):
        para = para.strip()
        if not para:
            continue
        if _LIST_ITEM.match(para):
            units.append(("list", para, None))
        else:
            for sent in _SPLIT_RE.split(para):
                if sent.strip():
                    units.append(("paragraph", sent.strip(), None))
    return units


def _force_split(text: str, btype: str, max_tokens: int) -> list[tuple[str, str]]:
    if count_tokens(text) <= max_tokens:
        return [(btype, text)]
    words = text.split()
    parts: list[tuple[str, str]] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if count_tokens(" ".join(cur)) >= max_tokens:
            parts.append((btype, " ".join(cur)))
            cur = []
    if cur:
        parts.append((btype, " ".join(cur)))
    return parts


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for u in reversed(units):
        t = count_tokens(u)
        if total + t > overlap and tail:
            break
        tail.insert(0, u)
        total += t
    return tail


def chunk_sections(
    sections: list[Section],
    *,
    document_type: str = "digital_pdf",
    source_language: str = "as",
    page_confidence: dict[int, float] | None = None,
) -> list[ChunkDraft]:
    target = settings.chunk_target_tokens
    max_t = settings.chunk_max_tokens
    overlap = settings.chunk_overlap_tokens
    page_confidence = page_confidence or {}

    drafts: list[ChunkDraft] = []
    index = 0

    for section in sections:
        raw_units = _units_from_section(section)
        units: list[tuple[str, str]] = []
        for btype, text, _ in raw_units:
            text, q = clean_text(text, preserve_line_breaks=(btype in ("qa", "list", "table")))
            if not is_embeddable(text, settings.unicode_quality_threshold):
                continue
            for utype, part in _force_split(text, btype, max_t):
                units.append((utype, part))

        current: list[str] = []
        current_types: list[str] = []
        current_tokens = 0

        def _emit(unit_list: list[str], types: list[str]):
            nonlocal index
            sep = "\n\n" if any(t in ("qa", "list", "table") for t in types) else " "
            text = sep.join(unit_list).strip()
            if not text:
                return
            q = quality_score(text)
            drafts.append(
                ChunkDraft(
                    chunk_index=index,
                    text=text,
                    section_id=section.section_id,
                    section_title=section.title,
                    heading_path=list(section.heading_path),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    block_type=types[0] if types else "paragraph",
                    token_count=count_tokens(text),
                    quality_score=q,
                    language=detect_language(text, default=source_language),
                    text_hash=_hash(text),
                    ocr_confidence=page_confidence.get(section.page_start),
                )
            )
            index += 1

        for btype, unit in units:
            ut = count_tokens(unit)
            if ut >= max_t:
                if current:
                    _emit(current, current_types)
                    current, current_types, current_tokens = [], [], 0
                _emit([unit], [btype])
                continue
            if current_tokens + ut > target and current:
                _emit(current, current_types)
                tail = _overlap_tail(current, overlap)
                current = list(tail)
                current_types = [current_types[-1]] * len(tail) if current_types else []
                current_tokens = sum(count_tokens(u) for u in current)
            current.append(unit)
            current_types.append(btype)
            current_tokens += ut

        if current:
            _emit(current, current_types)

    drafts = deduplicate_chunks(drafts)
    drafts = merge_tiny_chunks(drafts)
    drafts = filter_noise(drafts)
    # Re-index
    for i, d in enumerate(drafts):
        d.chunk_index = i
    log.info("chunker.semantic", chunks=len(drafts), document_type=document_type)
    return drafts

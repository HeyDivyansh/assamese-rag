"""Semantic chunking.

Rules (spec §3):
  * token counter = BGE-M3's own tokenizer (see embedding.count_tokens)
  * target ~400 tokens, overlap ~70 tokens
  * respect section boundaries (never merge across sections; never split a
    heading away from its body unnecessarily)
  * Q&A datasets: keep each question+answer pair together as a unit

We split each section into sentence-ish units, then greedily pack units up to
the target token budget, carrying an overlap tail into the next chunk.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.ingestion.cleaning import looks_like_qa_lines
from app.ingestion.embedding import count_tokens
from app.ingestion.sectioning import Section

# Split on sentence terminators incl. Bengali/Devanagari danda (।).
_SPLIT_RE = re.compile(r"(?<=[।\.\?\!])\s+|\n{2,}")


@dataclass
class ChunkDraft:
    chunk_index: int
    text: str
    section_id: uuid.UUID | None
    section_title: str | None
    page_number: int | None
    token_count: int


def _split_qa_units(text: str) -> list[str]:
    """One unit per question line + following answer line."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    units: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("?") or line.endswith("？"):
            answer = ""
            if i + 1 < len(lines) and not (
                lines[i + 1].endswith("?") or lines[i + 1].endswith("？")
            ):
                answer = lines[i + 1]
                i += 2
            else:
                i += 1
            units.append(f"{line}\n{answer}".strip() if answer else line)
        else:
            units.append(line)
            i += 1
    return units


def _split_units(text: str) -> list[str]:
    if looks_like_qa_lines(text):
        return _split_qa_units(text)
    units = [u.strip() for u in _SPLIT_RE.split(text) if u and u.strip()]
    return units


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing units whose combined token count ~ overlap_tokens."""
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        t = count_tokens(unit)
        if total + t > overlap_tokens and tail:
            break
        tail.insert(0, unit)
        total += t
    return tail


def chunk_sections(
    sections: list[Section],
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[ChunkDraft]:
    target = target_tokens or settings.chunk_target_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    drafts: list[ChunkDraft] = []
    index = 0

    for section in sections:
        body = section.text or ""
        if section.title:
            # Prepend heading so the chunk keeps its context; don't strand it.
            body = f"{section.title}\n{body}".strip()
        units = _split_units(body)
        if not units:
            continue

        current: list[str] = []
        current_tokens = 0

        def _emit(unit_list: list[str]):
            nonlocal index
            sep = "\n\n" if looks_like_qa_lines("\n".join(unit_list)) else " "
            text = sep.join(unit_list).strip()
            if not text:
                return
            drafts.append(
                ChunkDraft(
                    chunk_index=index,
                    text=text,
                    section_id=section.section_id,
                    section_title=section.title,
                    page_number=section.page_number,
                    token_count=count_tokens(text),
                )
            )
            index += 1

        for unit in units:
            ut = count_tokens(unit)
            # A single oversized unit becomes its own chunk.
            if ut >= target:
                if current:
                    _emit(current)
                    current, current_tokens = [], 0
                _emit([unit])
                continue
            if current_tokens + ut > target and current:
                _emit(current)
                tail = _overlap_tail(current, overlap)
                current = list(tail)
                current_tokens = sum(count_tokens(u) for u in current)
            current.append(unit)
            current_tokens += ut

        if current:
            _emit(current)

    return drafts

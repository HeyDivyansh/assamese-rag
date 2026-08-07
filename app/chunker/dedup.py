"""Duplicate removal and noise filtering (P0)."""
from __future__ import annotations

import hashlib

from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.types import ChunkDraft

log = get_logger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_chunks(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    seen: set[str] = set()
    out: list[ChunkDraft] = []
    for d in drafts:
        h = d.text_hash or _hash(d.text)
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    dropped = len(drafts) - len(out)
    if dropped:
        log.info("chunker.dedup", dropped=dropped, kept=len(out))
    return out


def merge_tiny_chunks(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    """Merge chunks below min token threshold into previous if same section."""
    if not drafts:
        return []
    min_t = settings.chunk_min_tokens
    out: list[ChunkDraft] = []
    for d in drafts:
        if (
            out
            and d.token_count < min_t
            and out[-1].section_id == d.section_id
            and out[-1].token_count + d.token_count <= settings.chunk_max_tokens
        ):
            prev = out[-1]
            merged_text = prev.text + "\n\n" + d.text
            out[-1] = ChunkDraft(
                chunk_index=prev.chunk_index,
                text=merged_text,
                section_id=prev.section_id,
                section_title=prev.section_title,
                heading_path=prev.heading_path,
                page_start=prev.page_start,
                page_end=max(prev.page_end or 0, d.page_end or 0),
                block_type=prev.block_type,
                token_count=prev.token_count + d.token_count,
                quality_score=min(prev.quality_score, d.quality_score),
                language=prev.language,
                text_hash=_hash(merged_text),
                ocr_confidence=prev.ocr_confidence,
            )
        else:
            out.append(d)
    return out


def filter_noise(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    out: list[ChunkDraft] = []
    for d in drafts:
        if len(d.text.strip()) < 10:
            continue
        if d.quality_score < settings.unicode_quality_threshold:
            continue
        if d.token_count > settings.chunk_max_tokens:
            continue
        out.append(d)
    dropped = len(drafts) - len(out)
    if dropped:
        log.info("chunker.filter_noise", dropped=dropped, kept=len(out))
    return out

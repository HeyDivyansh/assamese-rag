"""LLM context assembly (P3)."""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)


def _sort_key(c: RetrievedChunk) -> tuple:
    p = c.payload or {}
    return (
        p.get("page_start") or c.page_number or 0,
        p.get("chunk_index") or 0,
        str(c.section_id or ""),
    )


def build_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Sort by reading order, merge adjacent same-section chunks, dedupe text."""
    if not chunks:
        return []
    ordered = sorted(chunks, key=_sort_key)
    seen_text: set[str] = set()
    out: list[RetrievedChunk] = []
    max_tokens = settings.context_max_tokens
    total = 0

    for c in ordered:
        text = (c.text or "").strip()
        if not text or text in seen_text:
            continue
        tok = c.payload.get("token_count") or len(text.split())
        if total + tok > max_tokens:
            break
        seen_text.add(text)
        total += tok
        # Merge with previous if adjacent indices same section
        if (
            out
            and out[-1].section_id == c.section_id
            and out[-1].document_id == c.document_id
            and c.chunk_index is not None
            and out[-1].chunk_index is not None
            and c.chunk_index == out[-1].chunk_index + 1
            and total <= max_tokens
        ):
            prev = out[-1]
            merged = prev.text + "\n" + text
            prev.text = merged
            prev.payload["expanded_text"] = merged
            continue
        c.payload["expanded_text"] = text
        out.append(c)

    log.info("prompt_builder.context", chunks=len(out), approx_tokens=total)
    return out

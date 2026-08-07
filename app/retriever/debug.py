"""Retrieval debug logging (P3)."""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)


def log_retrieval_trace(
    query: str,
    *,
    embed_backend: str,
    dense_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    fused: list[RetrievedChunk],
    mmr: list[RetrievedChunk],
    reranked: list[RetrievedChunk],
    final: list[RetrievedChunk],
) -> None:
    if not settings.debug_retrieval:
        return

    def _rows(label: str, chunks: list[RetrievedChunk]):
        rows = []
        for i, c in enumerate(chunks, 1):
            preview = (c.text or "")[:300].replace("\n", " ")
            rows.append({
                "rank": i,
                "chunk_id": str(c.chunk_id),
                "score": round(c.score, 4),
                "section": c.section_title,
                "page": c.payload.get("page_start") or c.page_number,
                "preview": preview,
                "source": c.source,
            })
        return {label: rows}

    trace = {
        "query": query,
        "embed_backend": embed_backend,
        **_rows("dense", dense_hits),
        **_rows("bm25", bm25_hits),
        **_rows("fused", fused),
        **_rows("mmr", mmr),
        **_rows("reranked", reranked),
        **_rows("final", final),
    }
    log.info("retrieval.debug_trace", **trace)

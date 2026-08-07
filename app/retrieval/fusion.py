"""Reciprocal Rank Fusion (RRF) of dense + BM25 result lists.

RRF score for a doc = sum over lists of 1 / (k + rank), rank 1-indexed.
k defaults to 60 (settings.rrf_k). This is robust to the very different score
scales of cosine similarity vs BM25.
"""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    top_k: int | None = None,
    k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.rrf_top_k
    k = k or settings.rrf_k

    fused_scores: dict[str, float] = {}
    best_obj: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = str(item.chunk_id)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Prefer keeping the object that carries text/metadata.
            if key not in best_obj or (not best_obj[key].text and item.text):
                best_obj[key] = item

    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[RetrievedChunk] = []
    for key, score in ranked[:top_k]:
        obj = best_obj[key]
        obj.score = score
        obj.source = "fused"
        out.append(obj)
    log.info("fusion.rrf", inputs=len(result_lists), fused=len(out), top_k=top_k)
    return out

"""Maximal Marginal Relevance for diverse retrieval (P3)."""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)


def _token_set(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def maximal_marginal_relevance(
    candidates: list[RetrievedChunk],
    *,
    top_k: int | None = None,
    lambda_param: float | None = None,
) -> list[RetrievedChunk]:
    """Lexical MMR using RRF/rerank scores and token Jaccard diversity."""
    top_k = top_k or settings.mmr_top_k
    lam = lambda_param if lambda_param is not None else settings.mmr_lambda
    if len(candidates) <= top_k:
        return candidates

    selected: list[RetrievedChunk] = []
    remaining = list(candidates)
    token_sets = {str(c.chunk_id): _token_set(c.text) for c in candidates}
    max_score = max((c.score for c in candidates), default=1.0) or 1.0

    while remaining and len(selected) < top_k:
        best: RetrievedChunk | None = None
        best_mmr = -1.0
        for c in remaining:
            rel = c.score / max_score
            div = 0.0
            if selected:
                div = max(
                    _jaccard(token_sets[str(c.chunk_id)], token_sets[str(s.chunk_id)])
                    for s in selected
                )
            mmr = lam * rel - (1 - lam) * div
            if mmr > best_mmr:
                best_mmr = mmr
                best = c
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    log.info("retriever.mmr", in_len=len(candidates), out_len=len(selected))
    return selected

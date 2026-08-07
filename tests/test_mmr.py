"""Unit tests for lexical MMR (P3)."""
from __future__ import annotations

import uuid

from app.retriever.mmr import maximal_marginal_relevance
from app.retrieval.types import RetrievedChunk


def _chunk(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        score=score,
    )


def test_mmr_prefers_diverse_chunks():
    candidates = [
        _chunk("অসমীয়া ভাষা সম্পর্কে", 1.0),
        _chunk("অসমীয়া ভাষা সম্পর্কে বিস্তাৰিত", 0.95),  # near duplicate
        _chunk("গুৱাহাটীৰ ইতিহাস", 0.8),
    ]
    out = maximal_marginal_relevance(candidates, top_k=2, lambda_param=0.7)
    texts = {c.text for c in out}
    assert len(out) == 2
    assert "গুৱাহাটীৰ ইতিহাস" in texts


def test_mmr_returns_all_when_small():
    candidates = [_chunk("a", 1.0), _chunk("b", 0.5)]
    out = maximal_marginal_relevance(candidates, top_k=5)
    assert len(out) == 2

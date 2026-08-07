"""Unit tests for chunk deduplication (P0)."""
from __future__ import annotations

import uuid

from app.chunker.dedup import deduplicate_chunks, filter_noise, merge_tiny_chunks
from app.pipeline.types import ChunkDraft


def _draft(text: str, tokens: int = 100, quality: float = 0.9) -> ChunkDraft:
    sid = uuid.uuid4()
    return ChunkDraft(
        chunk_index=0,
        text=text,
        section_id=sid,
        section_title="Test",
        heading_path=["Test"],
        page_start=1,
        page_end=1,
        block_type="paragraph",
        token_count=tokens,
        quality_score=quality,
        language="as",
        text_hash="",
    )


def test_deduplicate_chunks():
    a = _draft("same text")
    b = _draft("same text")
    c = _draft("different")
    out = deduplicate_chunks([a, b, c])
    assert len(out) == 2


def test_merge_tiny_chunks_same_section():
    big = _draft("paragraph one", tokens=200)
    tiny = _draft("tiny", tokens=10)
    tiny.section_id = big.section_id
    out = merge_tiny_chunks([big, tiny])
    assert len(out) == 1
    assert "tiny" in out[0].text


def test_filter_noise_drops_low_quality():
    good = _draft("অসমীয়া পাঠ", quality=0.9)
    bad = _draft("Łł", tokens=5, quality=0.1)
    out = filter_noise([good, bad])
    assert len(out) == 1
    assert out[0].text == good.text

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A candidate chunk flowing through the retrieval pipeline."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    score: float = 0.0
    chunk_index: int | None = None
    section_id: uuid.UUID | None = None
    section_title: str | None = None
    page_number: int | None = None
    prev_chunk_id: uuid.UUID | None = None
    next_chunk_id: uuid.UUID | None = None
    source: str = ""  # 'dense' | 'bm25' | 'fused' | 'reranked'
    payload: dict = field(default_factory=dict)

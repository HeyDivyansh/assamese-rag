"""Context expansion.

For each reranked chunk, pull its prev/next neighbours so the LLM sees fuller
context than the bare matched window. Neighbours are fetched from Qdrant by
point id (we set the Qdrant point id == chunk id at ingestion, so prev/next
chunk ids are directly retrievable).
"""
from __future__ import annotations

import uuid

from app.core.clients import get_qdrant
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)


def _fetch_texts(point_ids: list[uuid.UUID]) -> dict[str, str]:
    if not point_ids:
        return {}
    client = get_qdrant()
    records = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[str(pid) for pid in point_ids],
        with_payload=True,
    )
    return {str(r.id): (r.payload or {}).get("text", "") for r in records}


def expand_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Attach `expanded_text` (prev + self + next) into each chunk's payload."""
    neighbour_ids: list[uuid.UUID] = []
    for c in chunks:
        if c.prev_chunk_id:
            neighbour_ids.append(c.prev_chunk_id)
        if c.next_chunk_id:
            neighbour_ids.append(c.next_chunk_id)

    texts = _fetch_texts(neighbour_ids)

    for c in chunks:
        parts: list[str] = []
        if c.prev_chunk_id and texts.get(str(c.prev_chunk_id)):
            parts.append(texts[str(c.prev_chunk_id)])
        parts.append(c.text)
        if c.next_chunk_id and texts.get(str(c.next_chunk_id)):
            parts.append(texts[str(c.next_chunk_id)])
        c.payload["expanded_text"] = "\n".join(p for p in parts if p)
    log.info("context_expansion.done", chunks=len(chunks), neighbours=len(texts))
    return chunks

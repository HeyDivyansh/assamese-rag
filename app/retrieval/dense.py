"""Dense retrieval + write path against Qdrant."""
from __future__ import annotations

import uuid

from qdrant_client.http import models as qmodels

from app.core.clients import get_qdrant
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)


def _as_uuid(v) -> uuid.UUID | None:
    if v is None:
        return None
    try:
        return uuid.UUID(str(v))
    except (ValueError, TypeError):
        return None


def upsert_chunks(points: list[dict]) -> None:
    """Write chunk vectors + payloads to Qdrant.

    `points` items: {id, vector, payload} where payload mirrors the metadata
    described in spec §7 (including the chunk `text`).
    """
    if not points:
        return
    client = get_qdrant()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qmodels.PointStruct(
                id=p["id"], vector=p["vector"], payload=p["payload"]
            )
            for p in points
        ],
    )
    log.info("qdrant.upsert", count=len(points))


def delete_by_document(document_id: uuid.UUID) -> None:
    client = get_qdrant()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=str(document_id)),
                    )
                ]
            )
        ),
    )
    log.info("qdrant.delete_by_document", document_id=str(document_id))


def _build_filter(
    user_id: uuid.UUID, document_ids: list[uuid.UUID] | None
) -> qmodels.Filter:
    must = [
        qmodels.FieldCondition(
            key="user_id", match=qmodels.MatchValue(value=str(user_id))
        )
    ]
    if document_ids:
        must.append(
            qmodels.FieldCondition(
                key="document_id",
                match=qmodels.MatchAny(any=[str(d) for d in document_ids]),
            )
        )
    return qmodels.Filter(must=must)


def dense_search(
    query_vector: list[float],
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    client = get_qdrant()
    top_k = top_k or settings.dense_top_k
    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=_build_filter(user_id, document_ids),
        limit=top_k,
        with_payload=True,
    )
    results: list[RetrievedChunk] = []
    for h in hits:
        p = h.payload or {}
        results.append(
            RetrievedChunk(
                chunk_id=_as_uuid(p.get("chunk_id")) or _as_uuid(h.id),
                document_id=_as_uuid(p.get("document_id")),
                text=p.get("text", ""),
                score=float(h.score),
                chunk_index=p.get("chunk_index"),
                section_id=_as_uuid(p.get("section_id")),
                section_title=p.get("section_title"),
                page_number=p.get("page_start") or p.get("page_number"),
                prev_chunk_id=_as_uuid(p.get("prev_chunk_id")),
                next_chunk_id=_as_uuid(p.get("next_chunk_id")),
                source="dense",
                payload={
                    **p,
                    "page_start": p.get("page_start") or p.get("page_number"),
                    "page_end": p.get("page_end") or p.get("page_number"),
                    "heading_path": p.get("heading_path"),
                    "block_type": p.get("block_type"),
                    "quality_score": p.get("quality_score"),
                    "document_type": p.get("document_type"),
                },
            )
        )
    log.info("qdrant.search", hits=len(results), top_k=top_k)
    return results

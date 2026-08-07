"""BM25 retrieval + write path against OpenSearch (Indic-aware analyzer)."""
from __future__ import annotations

import uuid

from app.core.clients import get_opensearch
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


def index_chunks(docs: list[dict]) -> None:
    """Bulk-index chunk text for BM25. Each doc must include `_id` + fields."""
    if not docs:
        return
    client = get_opensearch()
    bulk: list[dict] = []
    for d in docs:
        _id = d.pop("_id")
        bulk.append({"index": {"_index": settings.opensearch_index, "_id": _id}})
        bulk.append(d)
    client.bulk(body=bulk, refresh=True)
    log.info("opensearch.index", count=len(docs))


def delete_by_document(document_id: uuid.UUID) -> None:
    client = get_opensearch()
    client.delete_by_query(
        index=settings.opensearch_index,
        body={"query": {"term": {"document_id": str(document_id)}}},
        refresh=True,
        conflicts="proceed",
    )
    log.info("opensearch.delete_by_document", document_id=str(document_id))


def bm25_search(
    query_text: str,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    client = get_opensearch()
    top_k = top_k or settings.bm25_top_k

    filters: list[dict] = [{"term": {"user_id": str(user_id)}}]
    if document_ids:
        filters.append(
            {"terms": {"document_id": [str(d) for d in document_ids]}}
        )

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": {"match": {"text": {"query": query_text}}},
                "filter": filters,
            }
        },
    }
    resp = client.search(index=settings.opensearch_index, body=body)
    results: list[RetrievedChunk] = []
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        results.append(
            RetrievedChunk(
                chunk_id=_as_uuid(src.get("chunk_id")) or _as_uuid(h["_id"]),
                document_id=_as_uuid(src.get("document_id")),
                text=src.get("text", ""),
                score=float(h["_score"]),
                chunk_index=src.get("chunk_index"),
                section_id=_as_uuid(src.get("section_id")),
                section_title=src.get("section_title"),
                page_number=src.get("page_start") or src.get("page_number"),
                source="bm25",
                payload={
                    **src,
                    "page_start": src.get("page_start") or src.get("page_number"),
                    "page_end": src.get("page_end") or src.get("page_number"),
                },
            )
        )
    log.info("opensearch.search", hits=len(results), top_k=top_k)
    return results

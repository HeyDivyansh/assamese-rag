"""Factories + bootstrap for Qdrant and OpenSearch.

Kept dependency-light: the actual search logic lives in app/retrieval/*.
These helpers create clients and ensure the collection/index exist with the
right vector params and (crucially) an Indic-script-aware analyzer for BM25.
"""
from __future__ import annotations

from functools import lru_cache

from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
@lru_cache
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def ensure_qdrant_collection() -> None:
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return
    log.info("qdrant.create_collection", name=settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qmodels.VectorParams(
            size=settings.bge_m3_dim, distance=qmodels.Distance.COSINE
        ),
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
    )
    # Payload indexes for filters.
    for field in ("user_id", "document_id", "block_type", "document_type"):
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001
            log.warning("qdrant.payload_index_exists", field=field)


# --------------------------------------------------------------------------- #
# OpenSearch
# --------------------------------------------------------------------------- #
@lru_cache
def get_opensearch() -> OpenSearch:
    http_auth = None
    if settings.opensearch_user:
        http_auth = (settings.opensearch_user, settings.opensearch_password)
    use_ssl = settings.opensearch_url.startswith("https")
    return OpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=http_auth,
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
    )


# Assamese uses the Bengali/Eastern-Nagari script. The ICU analyzer (from the
# analysis-icu plugin) segments Indic scripts far better than the default
# English analyzer. We fall back to a unicode-normalizing custom analyzer if
# ICU is unavailable, but the default English analyzer is explicitly avoided.
_INDEX_BODY = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {
            "analyzer": {
                "assamese_icu": {
                    "type": "custom",
                    "tokenizer": "icu_tokenizer",
                    "char_filter": ["icu_normalizer"],
                    "filter": ["icu_folding"],
                },
                # Fallback if analysis-icu plugin is not installed.
                "assamese_fallback": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                },
            }
        },
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "section_id": {"type": "keyword"},
            "section_title": {"type": "text", "analyzer": "assamese_icu"},
            "page_number": {"type": "integer"},
            "page_start": {"type": "integer"},
            "page_end": {"type": "integer"},
            "block_type": {"type": "keyword"},
            "document_type": {"type": "keyword"},
            "heading_path": {"type": "keyword"},
            "quality_score": {"type": "float"},
            "text": {"type": "text", "analyzer": "assamese_icu"},
        }
    },
}


def ensure_opensearch_index() -> None:
    client = get_opensearch()
    if client.indices.exists(index=settings.opensearch_index):
        return
    log.info("opensearch.create_index", index=settings.opensearch_index)
    try:
        client.indices.create(index=settings.opensearch_index, body=_INDEX_BODY)
    except Exception:  # noqa: BLE001
        # Most likely the analysis-icu plugin is missing — retry with fallback.
        log.warning("opensearch.icu_unavailable_falling_back")
        body = {**_INDEX_BODY}
        body["mappings"]["properties"]["text"]["analyzer"] = "assamese_fallback"
        body["mappings"]["properties"]["section_title"]["analyzer"] = (
            "assamese_fallback"
        )
        client.indices.create(index=settings.opensearch_index, body=body)

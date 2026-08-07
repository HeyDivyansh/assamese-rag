"""End-to-end retrieval: embed → dense ‖ bm25 → RRF → MMR → rerank → context."""
from __future__ import annotations

import asyncio
import time
import uuid

from app.core.config import settings
from app.core.logging import get_logger
from app.core.stage_logger import astage, record_stage_async
from app.cleaner.unicode import clean_text
from app.ingestion.embedding import embed_query
from app.prompt_builder.context import build_context
from app.retrieval import bm25, dense
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import rerank
from app.retrieval.types import RetrievedChunk
from app.retriever.debug import log_retrieval_trace
from app.retriever.mmr import maximal_marginal_relevance

log = get_logger(__name__)


async def retrieve(
    db,
    query: str,
    user_id: uuid.UUID,
    *,
    document_ids: list[uuid.UUID] | None = None,
    conversation_id: uuid.UUID | None = None,
    request_id: str | uuid.UUID | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    cleaned, _ = clean_text(query)
    embed_component = "hf_inference" if settings.hf_token else "bge_m3"

    async with astage(db, "query_embedding", component=embed_component,
                      conversation_id=conversation_id, request_id=request_id) as st:
        qvec = await asyncio.to_thread(embed_query, cleaned)
        st.output({"dim": len(qvec), "backend": embed_component})

    async def _dense_search():
        t0 = time.perf_counter()
        res = await asyncio.to_thread(
            dense.dense_search, qvec, user_id, document_ids, settings.dense_top_k
        )
        return res, int((time.perf_counter() - t0) * 1000)

    async def _bm25_search():
        t0 = time.perf_counter()
        res = await asyncio.to_thread(
            bm25.bm25_search, cleaned, user_id, document_ids, settings.bm25_top_k
        )
        return res, int((time.perf_counter() - t0) * 1000)

    (dense_hits, dense_ms), (bm25_hits, bm25_ms) = await asyncio.gather(
        _dense_search(), _bm25_search()
    )

    await record_stage_async(
        db, "dense_retrieval", status="success", component="qdrant",
        duration_ms=dense_ms, conversation_id=conversation_id, request_id=request_id,
        output_summary={"hits": len(dense_hits)},
    )
    await record_stage_async(
        db, "bm25_retrieval", status="success", component="opensearch",
        duration_ms=bm25_ms, conversation_id=conversation_id, request_id=request_id,
        output_summary={"hits": len(bm25_hits)},
    )

    async with astage(db, "fusion", component="rrf",
                      conversation_id=conversation_id, request_id=request_id) as st:
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], top_k=settings.rrf_top_k)
        st.output({"candidates": len(fused)})

    if not fused:
        log_retrieval_trace(cleaned, embed_backend=embed_component,
                            dense_hits=dense_hits, bm25_hits=bm25_hits,
                            fused=[], mmr=[], reranked=[], final=[])
        return cleaned, []

    async with astage(db, "mmr", component="lexical_mmr",
                      conversation_id=conversation_id, request_id=request_id) as st:
        mmr_hits = await asyncio.to_thread(
            maximal_marginal_relevance, fused, top_k=settings.mmr_top_k
        )
        st.output({"candidates": len(mmr_hits)})

    rerank_input = mmr_hits[: settings.rerank_input_top_k]
    async with astage(db, "rerank", component="bge_reranker_v2_m3",
                      conversation_id=conversation_id, request_id=request_id) as st:
        reranked = await asyncio.to_thread(
            rerank, cleaned, rerank_input, settings.rerank_top_k
        )
        st.output({"kept": len(reranked)})

    async with astage(db, "context_build", component="prompt_builder",
                      conversation_id=conversation_id, request_id=request_id) as st:
        final = await asyncio.to_thread(build_context, reranked)
        st.output({"chunks": len(final)})

    log_retrieval_trace(
        cleaned,
        embed_backend=embed_component,
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        fused=fused,
        mmr=mmr_hits,
        reranked=reranked,
        final=final,
    )
    return cleaned, final

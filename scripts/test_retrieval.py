"""Standalone hybrid-retrieval smoke test (spec build-order step 5).

Validates dense + BM25 + RRF + rerank on Assamese text WITHOUT the full chat
pipeline, since hybrid retrieval quality on Assamese is the highest-risk part.

It seeds a handful of Assamese chunks directly into Qdrant + OpenSearch for a
throwaway user, runs each retrieval stage, and prints the ranked results.

Usage:
    python -m scripts.test_retrieval

Requires Qdrant + OpenSearch reachable (see .env) and the BGE-M3 model /
reranker available (local or endpoint).
"""
from __future__ import annotations

import uuid

from app.core.clients import ensure_opensearch_index, ensure_qdrant_collection
from app.ingestion.embedding import embed_texts, embed_query
from app.retrieval import bm25, dense
from app.prompt_builder.context import build_context
from app.retriever.mmr import maximal_marginal_relevance
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import rerank

USER_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()

# A few short Assamese passages on distinct topics.
CHUNKS = [
    "অসম ভাৰতৰ উত্তৰ-পূব অঞ্চলত অৱস্থিত এখন ৰাজ্য। ইয়াৰ ৰাজধানী দিছপুৰ।",
    "ব্ৰহ্মপুত্ৰ নদী অসমৰ মাজেৰে বৈ গৈছে আৰু ই কৃষিৰ বাবে অতি গুৰুত্বপূৰ্ণ।",
    "বিহু অসমৰ প্ৰধান উৎসৱ। বহাগ বিহু বসন্ত কালত উদযাপন কৰা হয়।",
    "অসমীয়া চাহ বিশ্ববিখ্যাত। অসমৰ চাহ বাগিচাবোৰে বহু মানুহক জীৱিকা দিয়ে।",
    "কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যান একশৃঙ্গী গঁড়ৰ বাবে বিখ্যাত।",
]

QUERY = "অসমৰ ৰাজধানী কি?"  # "What is the capital of Assam?"


def seed() -> list[uuid.UUID]:
    ensure_qdrant_collection()
    ensure_opensearch_index()

    ids = [uuid.uuid4() for _ in CHUNKS]
    vectors = embed_texts(CHUNKS)

    points, os_docs = [], []
    for i, (cid, text, vec) in enumerate(zip(ids, CHUNKS, vectors)):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None
        payload = {
            "chunk_id": str(cid),
            "document_id": str(DOC_ID),
            "user_id": str(USER_ID),
            "chunk_index": i,
            "section_title": None,
            "page_number": 1,
            "prev_chunk_id": str(prev_id) if prev_id else None,
            "next_chunk_id": str(next_id) if next_id else None,
            "text": text,
        }
        points.append({"id": str(cid), "vector": vec, "payload": payload})
        os_docs.append({
            "_id": str(cid), "chunk_id": str(cid), "document_id": str(DOC_ID),
            "user_id": str(USER_ID), "chunk_index": i, "page_number": 1, "text": text,
        })

    dense.upsert_chunks(points)
    bm25.index_chunks(os_docs)
    return ids


def _show(title: str, results):
    print(f"\n=== {title} ===")
    for r in results:
        print(f"  [{r.score:.4f}] ({r.source}) {r.text[:60]}")


def main() -> None:
    print(f"Seeding {len(CHUNKS)} Assamese chunks for user={USER_ID} ...")
    seed()

    print(f"\nQuery: {QUERY}")
    qvec = embed_query(QUERY)

    dense_hits = dense.dense_search(qvec, USER_ID, [DOC_ID], top_k=5)
    _show("DENSE (Qdrant)", dense_hits)

    bm25_hits = bm25.bm25_search(QUERY, USER_ID, [DOC_ID], top_k=5)
    _show("BM25 (OpenSearch)", bm25_hits)

    fused = reciprocal_rank_fusion([dense_hits, bm25_hits], top_k=5)
    _show("FUSED (RRF)", fused)

    mmr_hits = maximal_marginal_relevance(fused, top_k=5)
    _show("MMR (lexical diversity)", mmr_hits)

    reranked = rerank(QUERY, mmr_hits, top_k=3)
    _show("RERANKED (bge-reranker-v2-m3)", reranked)

    context = build_context(reranked)
    print("\n=== CONTEXT BUILDER (top-1 expanded_text) ===")
    if context:
        print(context[0].payload.get("expanded_text", "")[:200])


if __name__ == "__main__":
    main()

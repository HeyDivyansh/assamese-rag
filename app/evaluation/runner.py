"""Retrieval evaluation harness (P3/P5 stub)."""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class EvalCase:
    query: str
    expected_chunk_ids: list[str]
    document_ids: list[str] | None = None


@dataclass
class EvalMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    hit_rate: float
    latency_ms: float


def _recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return len(top & expected) / len(expected)


def _mrr(retrieved: list[str], expected: set[str]) -> float:
    for i, cid in enumerate(retrieved, 1):
        if cid in expected:
            return 1.0 / i
    return 0.0


def evaluate_cases(cases: list[EvalCase], retrieve_fn) -> EvalMetrics:
    """retrieve_fn(query, document_ids) -> list of chunk_id strings."""
    r5 = r10 = mrr_sum = hits = 0.0
    latencies: list[float] = []
    n = max(len(cases), 1)
    for case in cases:
        t0 = time.perf_counter()
        retrieved = retrieve_fn(case.query, case.document_ids)
        latencies.append((time.perf_counter() - t0) * 1000)
        expected = set(case.expected_chunk_ids)
        r5 += _recall_at_k(retrieved, expected, 5)
        r10 += _recall_at_k(retrieved, expected, 10)
        mrr_sum += _mrr(retrieved, expected)
        if expected & set(retrieved[:10]):
            hits += 1
    return EvalMetrics(
        recall_at_5=r5 / n,
        recall_at_10=r10 / n,
        mrr=mrr_sum / n,
        hit_rate=hits / n,
        latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            query=item["query"],
            expected_chunk_ids=item["expected_chunk_ids"],
            document_ids=item.get("document_ids"),
        )
        for item in raw
    ]


def run_benchmark(path: str | Path, retrieve_fn) -> EvalMetrics:
    cases = load_cases(path)
    metrics = evaluate_cases(cases, retrieve_fn)
    log.info(
        "evaluation.done",
        recall_at_5=round(metrics.recall_at_5, 3),
        recall_at_10=round(metrics.recall_at_10, 3),
        mrr=round(metrics.mrr, 3),
        hit_rate=round(metrics.hit_rate, 3),
        latency_ms=round(metrics.latency_ms, 1),
    )
    return metrics

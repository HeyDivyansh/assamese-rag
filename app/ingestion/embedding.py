"""BGE-M3 embeddings + tokenizer for chunk sizing.

Backends (first match wins):
  1. HF_TOKEN set          -> Hugging Face Inference API (no local model weights)
  2. BGE_M3_* is http URL  -> custom /embed endpoint
  3. otherwise             -> local sentence-transformers (dev fallback only)

The tokenizer is BGE-M3's own for chunk token counting (small vocab files only,
not the ~2 GB embedding weights).
"""
from __future__ import annotations

import math
import threading
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_model = None
_tokenizer = None
_hf_client = None


def _model_id() -> str:
    return settings.bge_m3_model_path_or_endpoint


def _use_hf_api() -> bool:
    return bool(settings.hf_token)


def _is_custom_endpoint() -> bool:
    return settings.bge_m3_model_path_or_endpoint.startswith("http")


def _embedding_backend() -> str:
    if _is_custom_endpoint():
        return "custom_http"
    if _use_hf_api():
        return "hf_inference"
    return "local"


def _get_hf_client():
    global _hf_client
    if _hf_client is not None:
        return _hf_client
    with _lock:
        if _hf_client is None:
            from huggingface_hub import InferenceClient

            log.info(
                "bge_m3.hf_client",
                model=_model_id(),
                provider="hf-inference",
            )
            _hf_client = InferenceClient(
                provider="hf-inference",
                api_key=settings.hf_token,
            )
    return _hf_client


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            path = _model_id()
            log.info("bge_m3.load", path=path, backend="sentence_transformers")
            _model = SentenceTransformer(path)
    return _model


def get_tokenizer():
    """BGE-M3 tokenizer (vocab only) for chunk token counting."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _lock:
        if _tokenizer is None:
            from transformers import AutoTokenizer

            name = _model_id() if not _is_custom_endpoint() else "BAAI/bge-m3"
            _tokenizer = AutoTokenizer.from_pretrained(name)
    return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def release_tokenizer() -> None:
    """Drop cached tokenizer to free RAM before a long embedding run."""
    global _tokenizer
    _tokenizer = None


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _to_vector(raw: Any) -> list[float]:
    """Convert HF feature_extraction output to a single L2-normalized vector."""
    import numpy as np

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr.mean(axis=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {arr.shape}")
    return _l2_normalize(vec.tolist())


def _embed_one_hf(text: str) -> list[float]:
    from huggingface_hub.errors import HfHubHTTPError

    client = _get_hf_client()
    try:
        raw = client.feature_extraction(text, model=_model_id())
    except HfHubHTTPError as exc:
        log.error(
            "bge_m3.hf_error",
            status=exc.response.status_code if exc.response else None,
            detail=str(exc)[:500],
        )
        raise
    return _to_vector(raw)


def _embed_batch_hf_parallel(texts: list[str]) -> list[list[float]]:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub.errors import HfHubHTTPError

    workers = max(1, settings.embedding_parallel_workers)
    results: list[list[float] | None] = [None] * len(texts)

    def _one(idx: int, text: str) -> tuple[int, list[float]]:
        for attempt in range(10):
            try:
                return idx, _embed_one_hf(text)
            except HfHubHTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    wait = min(30.0, 2.0 ** attempt)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("HF inference unavailable after retries (503)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i, t) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            idx, vec = fut.result()
            results[idx] = vec
    if any(v is None for v in results):
        raise RuntimeError("HF parallel embedding returned incomplete results")
    return results  # type: ignore[return-value]


def _embed_batch_hf(texts: list[str]) -> list[list[float]]:
    if settings.embedding_parallel_workers > 1 and len(texts) > 1:
        return _embed_batch_hf_parallel(texts)
    import time

    from huggingface_hub.errors import HfHubHTTPError

    out: list[list[float]] = []
    for text in texts:
        for attempt in range(10):
            try:
                out.append(_embed_one_hf(text))
                break
            except HfHubHTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    wait = min(30.0, 2.0 ** attempt)
                    time.sleep(wait)
                    continue
                raise
        else:
            raise RuntimeError("HF inference unavailable after retries (503)")
    return out


def _embed_batch_remote(texts: list[str]) -> list[list[float]]:
    url = settings.bge_m3_model_path_or_endpoint.rstrip("/") + "/embed"
    resp = httpx.post(url, json={"texts": texts}, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _embed_batch_local(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    backend = _embedding_backend()
    if backend == "hf_inference":
        return _embed_batch_hf(texts)
    if backend == "custom_http":
        return _embed_batch_remote(texts)
    return _embed_batch_local(texts)


def embed_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Return dense embeddings (list of float vectors) for the given texts."""
    if not texts:
        return []

    size = batch_size or settings.embedding_batch_size
    size = max(1, size)
    total = len(texts)
    out: list[list[float]] = []

    for start in range(0, total, size):
        batch = texts[start : start + size]
        batch_vecs = _embed_batch(batch)
        out.extend(batch_vecs)
        if total > size:
            log.info(
                "bge_m3.embed_batch",
                backend=_embedding_backend(),
                batch_start=start,
                batch_size=len(batch),
                total=total,
            )

    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text], batch_size=1)[0]

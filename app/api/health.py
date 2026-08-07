"""Liveness + readiness probes (spec §5 Ops)."""
from __future__ import annotations

import redis
from fastapi import APIRouter
from sqlalchemy import text

from app.core.clients import get_opensearch, get_qdrant
from app.core.config import settings
from app.core.db import async_engine
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz():
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe (deps connectivity)")
async def readyz():
    checks: dict[str, str] = {}
    ok = True

    # Postgres
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {exc}"
        ok = False

    # Qdrant
    try:
        get_qdrant().get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = f"error: {exc}"
        ok = False

    # OpenSearch
    try:
        get_opensearch().ping()
        checks["opensearch"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["opensearch"] = f"error: {exc}"
        ok = False

    # Redis
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"
        ok = False

    return {"status": "ready" if ok else "degraded", "checks": checks}

"""Helpers to record `pipeline_stage_logs` rows around each pipeline stage.

Usage (sync, ingestion worker)::

    with stage(db, "ocr", component="paddleocr", document_id=doc_id) as st:
        result = do_ocr(...)
        st.output({"pages": len(result)})

Usage (async, query pipeline)::

    async with astage(db, "dense_retrieval", component="qdrant",
                      conversation_id=cid) as st:
        hits = await dense_search(...)
        st.output({"hits": len(hits)})

On entry a `started` row is written; on clean exit it is updated to `success`
with duration; on exception a `failed` row is written with the error, and the
exception re-raised. This guarantees logging rule #4: a failure is traceable
end-to-end by request_id alone.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import update

from app.core.context import get_request_id
from app.core.logging import get_logger
from app.models import PipelineStageLog

log = get_logger(__name__)


class _StageHandle:
    def __init__(self) -> None:
        self._input: dict | None = None
        self._output: dict | None = None

    def input(self, summary: dict) -> None:
        self._input = summary

    def output(self, summary: dict) -> None:
        self._output = summary


def _coerce_request_id(request_id: str | uuid.UUID | None) -> uuid.UUID:
    rid = request_id or get_request_id() or str(uuid.uuid4())
    return uuid.UUID(str(rid))


def record_stage_sync(
    db,
    stage_name: str,
    *,
    status: str,
    component: str | None = None,
    duration_ms: int | None = None,
    document_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    request_id: str | uuid.UUID | None = None,
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    error_message: str | None = None,
) -> None:
    """Write a single completed pipeline_stage_logs row (sync).

    Used for per-attempt logging of retried external calls (logging rule #5),
    where a context manager around the whole call is too coarse.
    """
    db.add(
        PipelineStageLog(
            request_id=_coerce_request_id(request_id),
            document_id=document_id,
            conversation_id=conversation_id,
            stage_name=stage_name,
            component=component,
            status=status,
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            error_message=(error_message or None) and str(error_message)[:2000],
        )
    )
    db.flush()


async def record_stage_async(
    db,
    stage_name: str,
    *,
    status: str,
    component: str | None = None,
    duration_ms: int | None = None,
    document_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    request_id: str | uuid.UUID | None = None,
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    error_message: str | None = None,
) -> None:
    """Write a single completed pipeline_stage_logs row (async)."""
    db.add(
        PipelineStageLog(
            request_id=_coerce_request_id(request_id),
            document_id=document_id,
            conversation_id=conversation_id,
            stage_name=stage_name,
            component=component,
            status=status,
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            error_message=(error_message or None) and str(error_message)[:2000],
        )
    )
    await db.flush()


@contextmanager
def stage(
    db,
    stage_name: str,
    *,
    component: str | None = None,
    document_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    request_id: str | uuid.UUID | None = None,
):
    """Sync stage context manager (workers)."""
    rid = _coerce_request_id(request_id)
    handle = _StageHandle()
    row = PipelineStageLog(
        request_id=rid,
        document_id=document_id,
        conversation_id=conversation_id,
        stage_name=stage_name,
        component=component,
        status="started",
    )
    db.add(row)
    db.flush()  # get row.id
    started = time.perf_counter()
    log.info("stage.start", stage=stage_name, component=component)
    try:
        yield handle
    except Exception as exc:  # noqa: BLE001
        dur = int((time.perf_counter() - started) * 1000)
        db.execute(
            update(PipelineStageLog)
            .where(PipelineStageLog.id == row.id)
            .values(
                status="failed",
                duration_ms=dur,
                input_summary=handle._input,
                output_summary=handle._output,
                error_message=str(exc)[:2000],
            )
        )
        db.flush()
        log.error(
            "stage.failed", stage=stage_name, component=component,
            duration_ms=dur, error=str(exc),
        )
        raise
    else:
        dur = int((time.perf_counter() - started) * 1000)
        db.execute(
            update(PipelineStageLog)
            .where(PipelineStageLog.id == row.id)
            .values(
                status="success",
                duration_ms=dur,
                input_summary=handle._input,
                output_summary=handle._output,
            )
        )
        db.flush()
        log.info(
            "stage.success", stage=stage_name, component=component, duration_ms=dur
        )


@asynccontextmanager
async def astage(
    db,
    stage_name: str,
    *,
    component: str | None = None,
    document_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    request_id: str | uuid.UUID | None = None,
):
    """Async stage context manager (query pipeline)."""
    rid = _coerce_request_id(request_id)
    handle = _StageHandle()
    row = PipelineStageLog(
        request_id=rid,
        document_id=document_id,
        conversation_id=conversation_id,
        stage_name=stage_name,
        component=component,
        status="started",
    )
    db.add(row)
    await db.flush()
    started = time.perf_counter()
    log.info("stage.start", stage=stage_name, component=component)
    try:
        yield handle
    except Exception as exc:  # noqa: BLE001
        dur = int((time.perf_counter() - started) * 1000)
        await db.execute(
            update(PipelineStageLog)
            .where(PipelineStageLog.id == row.id)
            .values(
                status="failed",
                duration_ms=dur,
                input_summary=handle._input,
                output_summary=handle._output,
                error_message=str(exc)[:2000],
            )
        )
        await db.flush()
        log.error(
            "stage.failed", stage=stage_name, component=component,
            duration_ms=dur, error=str(exc),
        )
        raise
    else:
        dur = int((time.perf_counter() - started) * 1000)
        await db.execute(
            update(PipelineStageLog)
            .where(PipelineStageLog.id == row.id)
            .values(
                status="success",
                duration_ms=dur,
                input_summary=handle._input,
                output_summary=handle._output,
            )
        )
        await db.flush()
        log.info(
            "stage.success", stage=stage_name, component=component, duration_ms=dur
        )

"""Per-request correlation context.

`request_id` and `user_id` are stored in context vars so that any log line or
downstream call anywhere in the stack (including deep inside the pipelines) can
attach them without threading arguments everywhere. Celery jobs re-seed these
from the arguments they receive.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def set_request_id(value: str | None) -> str:
    rid = value or str(uuid.uuid4())
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    return _request_id.get()


def set_user_id(value: str | None) -> None:
    _user_id.set(value)


def get_user_id() -> str | None:
    return _user_id.get()


def new_job_request_id() -> str:
    """A stable correlation id for an async ingestion job."""
    return str(uuid.uuid4())

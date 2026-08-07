"""Structured JSON logging via structlog.

Every emitted line carries `timestamp, level, request_id, user_id` (from the
context vars) plus whatever kwargs the caller passes (stage, message,
duration_ms, ...). This satisfies logging rule #3 in the spec.
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import settings
from app.core.context import get_request_id, get_user_id


def _inject_context(_, __, event_dict: dict) -> dict:
    """structlog processor: attach correlation ids to every log line."""
    rid = get_request_id()
    uid = get_user_id()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    if uid is not None:
        event_dict.setdefault("user_id", uid)
    return event_dict


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _inject_context,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

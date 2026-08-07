"""Celery application for the async ingestion pipeline."""
from __future__ import annotations

from celery import Celery

from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

celery_app = Celery(
    "assamese_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # OCR/embedding are heavy; one at a time.
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60,
)

# Ensure task modules are imported so Celery registers them.
celery_app.autodiscover_tasks(["app.workers"])

from app.workers import tasks  # noqa: E402,F401

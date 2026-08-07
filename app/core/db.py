"""Database engine + session management.

Two flavours are exposed:
  * async engine/session — used by FastAPI request handlers.
  * sync engine/session  — used by Celery workers (Celery + async is painful;
    the ingestion pipeline is CPU/IO heavy and runs fine on sync SQLAlchemy).
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def _to_async_url(url: str) -> str:
    # psycopg (v3) supports both sync & async via the same driver name.
    return url


def _to_sync_url(url: str) -> str:
    return url


# --- Async (API) ---
async_engine = create_async_engine(
    _to_async_url(settings.postgres_url),
    pool_pre_ping=True,
    future=True,
)
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# --- Sync (workers) ---
sync_engine = create_engine(
    _to_sync_url(settings.postgres_url),
    pool_pre_ping=True,
    future=True,
)
SyncSessionLocal = sessionmaker(bind=sync_engine, class_=Session, expire_on_commit=False)


@contextmanager
def get_sync_db():
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

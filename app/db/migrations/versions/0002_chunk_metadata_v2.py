"""chunk metadata v2 — P0/P2 schema extensions

Revision ID: 0002_chunk_metadata_v2
Revises: 0001_initial
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_chunk_metadata_v2"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("document_type", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("heading_path", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("chunks", sa.Column("block_type", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("quality_score", sa.Float(), nullable=True))
    op.add_column("chunks", sa.Column("language", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("text", sa.Text(), nullable=True))
    # Backfill page_start from page_number
    op.execute("UPDATE chunks SET page_start = page_number WHERE page_start IS NULL")


def downgrade() -> None:
    op.drop_column("chunks", "text")
    op.drop_column("chunks", "language")
    op.drop_column("chunks", "quality_score")
    op.drop_column("chunks", "block_type")
    op.drop_column("chunks", "heading_path")
    op.drop_column("chunks", "page_end")
    op.drop_column("chunks", "page_start")
    op.drop_column("documents", "document_type")

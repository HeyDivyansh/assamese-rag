"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-06

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # gen_random_uuid() lives in pgcrypto on some PG builds.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("s3_bucket", sa.Text(), nullable=False),
        sa.Column("s3_key", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger()),
        sa.Column("page_count", sa.Integer()),
        sa.Column("language", sa.Text(), server_default="as"),
        sa.Column("status", sa.Text(), nullable=False, server_default="uploaded"),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_documents_user_id", "documents", ["user_id"],
                    postgresql_where=sa.text("deleted_at IS NULL"))

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True)),
        sa.Column("section_title", sa.Text()),
        sa.Column("page_number", sa.Integer()),
        sa.Column("token_count", sa.Integer()),
        sa.Column("prev_chunk_id", postgresql.UUID(as_uuid=True)),
        sa.Column("next_chunk_id", postgresql.UUID(as_uuid=True)),
        sa.Column("ocr_confidence", sa.Float()),
        sa.Column("qdrant_point_id", postgresql.UUID(as_uuid=True)),
        sa.Column("opensearch_doc_id", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("idx_chunks_document_id", "chunks", ["document_id"])

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("last_message_at", sa.DateTime(timezone=True)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_conversations_user_id", "conversations", ["user_id"],
                    postgresql_where=sa.text("deleted_at IS NULL"))

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("input_type", sa.Text(), nullable=False, server_default="text"),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("raw_audio_s3_key", sa.Text()),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB()),
        sa.Column("model_used", sa.Text()),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint("role IN ('user','assistant')", name="ck_messages_role"),
        sa.CheckConstraint("input_type IN ('text','voice')",
                           name="ck_messages_input_type"),
    )
    op.create_index("idx_messages_conversation_id", "messages",
                    ["conversation_id", "created_at"])

    op.create_table(
        "api_request_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True)),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("status_code", sa.Integer()),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("request_summary", postgresql.JSONB()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("idx_api_request_logs_request_id", "api_request_logs",
                    ["request_id"])
    op.create_index("idx_api_request_logs_user_id_created_at", "api_request_logs",
                    ["user_id", "created_at"])

    op.create_table(
        "pipeline_stage_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True)),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True)),
        sa.Column("stage_name", sa.Text(), nullable=False),
        sa.Column("component", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column("input_summary", postgresql.JSONB()),
        sa.Column("output_summary", postgresql.JSONB()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint("status IN ('started','success','failed')",
                           name="ck_pipeline_stage_logs_status"),
    )
    op.create_index("idx_pipeline_stage_logs_request_id", "pipeline_stage_logs",
                    ["request_id"])
    op.create_index("idx_pipeline_stage_logs_document_id", "pipeline_stage_logs",
                    ["document_id"])


def downgrade() -> None:
    op.drop_table("pipeline_stage_logs")
    op.drop_table("api_request_logs")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("chunks")
    op.drop_table("documents")

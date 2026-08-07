from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    status: str


class DocumentOut(BaseModel):
    id: uuid.UUID
    original_filename: str
    display_name: str
    mime_type: str
    file_size_bytes: int | None = None
    page_count: int | None = None
    language: str
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class DocumentRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)


class DeleteResponse(BaseModel):
    id: uuid.UUID
    deleted: bool = True

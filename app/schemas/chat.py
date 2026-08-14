from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    section_title: str | None = None
    score: float | None = None
    preview: str | None = None


class ChatTextRequest(BaseModel):
    conversation_id: uuid.UUID | None = None
    message: str = Field(min_length=1)
    document_ids: list[uuid.UUID] | None = None


class ChatTextResponse(BaseModel):
    conversation_id: uuid.UUID
    answer: str
    sources: list[SourceRef] = []


class ChatVoiceResponse(BaseModel):
    conversation_id: uuid.UUID
    transcript: str
    answer: str
    sources: list[SourceRef] = []
    audio_base64: str | None = None


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    items: list[ConversationOut]
    total: int
    limit: int
    offset: int


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    input_type: str
    content_text: str
    retrieved_chunk_ids: list | dict | None = None
    model_used: str | None = None
    latency_ms: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    items: list[MessageOut]
    total: int
    limit: int
    offset: int

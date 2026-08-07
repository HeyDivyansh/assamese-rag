"""Conversation + message history endpoints (spec §5 Conversations)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.core.logging import get_logger
from app.models import Conversation, Message
from app.schemas.chat import (
    ConversationListResponse,
    ConversationOut,
    MessageListResponse,
    MessageOut,
)
from app.schemas.documents import DeleteResponse

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


async def _get_owned_conversation(
    db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation:
    conv = (
        await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("", response_model=ConversationListResponse, summary="List conversations")
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    base = select(Conversation).where(
        Conversation.user_id == user_id, Conversation.deleted_at.is_(None)
    )
    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await db.execute(
            base.order_by(
                func.coalesce(
                    Conversation.last_message_at, Conversation.created_at
                ).desc()
            )
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()
    return ConversationListResponse(
        items=[ConversationOut.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=MessageListResponse,
    summary="Get conversation messages (paginated)",
)
async def list_messages(
    conversation_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    await _get_owned_conversation(db, conversation_id, user_id)
    base = select(Message).where(Message.conversation_id == conversation_id)
    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await db.execute(
            base.order_by(Message.created_at.asc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return MessageListResponse(
        items=[MessageOut.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/{conversation_id}",
    response_model=DeleteResponse,
    summary="Soft-delete a conversation",
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    await _get_owned_conversation(db, conversation_id, user_id)
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(deleted_at=func.now())
    )
    log.info("conversation.deleted", conversation_id=str(conversation_id))
    return DeleteResponse(id=conversation_id, deleted=True)

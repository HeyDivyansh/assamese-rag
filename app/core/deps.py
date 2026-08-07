"""FastAPI dependencies for extracting the trusted identity/correlation ids.

The middleware already enforces presence; these just surface typed values to
handlers and keep OpenAPI/Swagger documenting the required headers.
"""
from __future__ import annotations

import uuid

from fastapi import Header, HTTPException, Request


async def get_current_user_id(
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> uuid.UUID:
    value = x_user_id or getattr(request.state, "user_id", None)
    if not value:
        raise HTTPException(status_code=401, detail="Missing required header: X-User-Id")
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="X-User-Id must be a UUID") from exc


async def get_request_id_dep(
    request: Request,
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> str:
    return getattr(request.state, "request_id", None) or x_request_id or str(
        uuid.uuid4()
    )

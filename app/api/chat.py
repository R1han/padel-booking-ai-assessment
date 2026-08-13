"""Streaming chat. Filled in once the agent graph exists."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["chat"])

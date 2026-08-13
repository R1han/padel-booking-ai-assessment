"""Slot read-back. Status is derived from claims, so it reflects bookings made at runtime."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.services import booking

router = APIRouter(prefix="/api/v1", tags=["slots"])


@router.get("/slots/{slot_id}")
def get_slot(slot_id: str, response: Response) -> dict:
    state = booking.slot_state(slot_id)
    if state is None:
        response.status_code = 404
        return {"error": "not_found", "message": f"No slot {slot_id}."}
    return state

"""Booking endpoints. Thin: they map service results onto status codes and nothing else.

Handlers are plain `def`, so Starlette runs them in the threadpool. An `async def` here
would block the event loop on sqlite and turn write contention into 20 timeouts.

Paths carry no trailing slash -- Starlette would 307-redirect the slashless POST, and the
race script does not follow redirects.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from app.services import booking

router = APIRouter(prefix="/api/v1", tags=["bookings"])


class BookingRequest(BaseModel):
    slot_ids: list[str] = Field(min_length=1)
    # Free-form on purpose: the race script invents user ids that are not in the dataset.
    user_id: str
    duration_min: int
    session_id: str | None = None
    idempotency_key: str | None = None


class HoldRequest(BaseModel):
    slot_ids: list[str] = Field(min_length=1)
    duration_min: int
    session_id: str


def _error(response: Response, exc: booking.BookingError) -> dict:
    response.status_code = exc.status
    return {"error": exc.error, "message": exc.message, "slot_ids": exc.slot_ids}


@router.post("/bookings", status_code=201)
def create_booking(req: BookingRequest, response: Response) -> dict:
    try:
        result = booking.create_booking(
            slot_ids=req.slot_ids,
            user_id=req.user_id,
            duration_min=req.duration_min,
            session_id=req.session_id,
            idempotency_key=req.idempotency_key,
        )
    except booking.BookingError as exc:
        return _error(response, exc)
    return {
        "booking_id": result.booking_id,
        "status": result.status,
        "slot_ids": result.slot_ids,
        "price_aed": result.price_aed,
    }


@router.post("/holds", status_code=201)
def create_hold(req: HoldRequest, response: Response) -> dict:
    try:
        result = booking.create_hold(req.slot_ids, req.duration_min, req.session_id)
    except booking.BookingError as exc:
        return _error(response, exc)
    return {
        "hold_id": result.hold_id,
        "slot_ids": result.slot_ids,
        "expires_at": result.expires_at,
    }


@router.delete("/holds/{hold_id}")
def release_hold(hold_id: str) -> dict:
    return {"released": booking.release_hold(hold_id)}


@router.delete("/bookings/{booking_id}")
def cancel_booking(booking_id: str, response: Response) -> dict:
    if not booking.cancel_booking(booking_id):
        response.status_code = 404
        return {"error": "not_found", "message": f"No booking {booking_id}."}
    return {"booking_id": booking_id, "status": "cancelled"}

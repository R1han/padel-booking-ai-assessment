"""Booking correctness, especially under concurrency.

The contract's own pass condition is the headline test: 20 simultaneous requests for one
slot must yield exactly one confirmation and no 5xx.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from app import db
from app.services import booking

# 06:00 on the first day of the window, on a court whose neighbour hour is also clear.
FREE = "slot_alquoz_pc02_20260812_0600"
FREE_NEXT = "slot_alquoz_pc02_20260812_0700"
# The grid runs 06-10 then 15-23. Nothing follows 10:00, so 90 minutes cannot fit.
GAP_EDGE = "slot_alquoz_pc02_20260812_1000"
DAY_END = "slot_alquoz_pc02_20260812_2300"


UNCLAIMED = (
    " id NOT IN (SELECT slot_id FROM slot_claims)"
    " AND id NOT IN (SELECT slot_id FROM slot_overhang)"
)


def _free_slot() -> str:
    """The first slot with no claim and no legacy overhang. Tests book as they go, so each
    call returns a fresh one without needing offsets."""
    with db.read_conn() as conn:
        row = conn.execute(
            f"SELECT id FROM slots WHERE status='available' AND {UNCLAIMED} ORDER BY id LIMIT 1"
        ).fetchone()
    return row["id"]


def _free_pair() -> str:
    """The first slot whose next real hour on the same court is also free -- required for
    any 90 or 120 minute booking."""
    with db.read_conn() as conn:
        row = conn.execute(
            """SELECT s.id FROM slots s
                JOIN slots n ON n.court_id = s.court_id AND n.date = s.date
                    AND n.start_time = printf('%02d:00', CAST(substr(s.start_time,1,2) AS INT) + 1)
                WHERE s.status='available' AND n.status='available'
                  AND s.id NOT IN (SELECT slot_id FROM slot_claims)
                  AND s.id NOT IN (SELECT slot_id FROM slot_overhang)
                  AND n.id NOT IN (SELECT slot_id FROM slot_claims)
                  AND n.id NOT IN (SELECT slot_id FROM slot_overhang)
                ORDER BY s.id LIMIT 1"""
        ).fetchone()
    return row["id"]


# --- the graded case --------------------------------------------------------------


def test_twenty_concurrent_bookings_yield_exactly_one_confirmation():
    slot = _free_slot()
    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(
            pool.map(
                lambda i: _try(slot, f"usr_race_{i}"),
                range(20),
            )
        )
    assert outcomes.count("ok") == 1, outcomes
    assert outcomes.count("conflict") == 19, outcomes
    assert "error" not in outcomes, outcomes


def _try(slot: str, user: str) -> str:
    try:
        booking.create_booking([slot], user, 60)
        return "ok"
    except booking.SlotUnavailable:
        return "conflict"
    except Exception:  # anything else would be a 5xx, which fails the contract
        return "error"


def test_slot_reads_back_as_exactly_one_booking_after_the_race():
    slot = _free_slot()
    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(lambda i: _try(slot, f"usr_{i}"), range(20)))
    state = booking.slot_state(slot)
    assert state["status"] == "booked"
    assert len(state["bookings"]) == 1


# --- duration and adjacency -------------------------------------------------------


def test_ninety_minutes_claims_two_contiguous_slots():
    slot = _free_pair()
    result = booking.create_booking([slot], "u", 90)
    assert len(result.slot_ids) == 2
    with db.read_conn() as conn:
        rows = [
            dict(conn.execute("SELECT court_id, date, start_time FROM slots WHERE id=?", (s,)).fetchone())
            for s in result.slot_ids
        ]
    assert rows[0]["court_id"] == rows[1]["court_id"]
    assert rows[0]["date"] == rows[1]["date"]
    assert int(rows[1]["start_time"][:2]) - int(rows[0]["start_time"][:2]) == 1


def test_ninety_minutes_is_priced_at_one_and_a_half_slots():
    slot = _free_pair()
    with db.read_conn() as conn:
        hourly = conn.execute("SELECT price_aed FROM slots WHERE id=?", (slot,)).fetchone()[0]
    result = booking.create_booking([slot], "u", 90)
    assert result.price_aed == round(hourly * 1.5)


def test_ninety_minutes_rejected_across_the_midday_gap():
    """10:00's neighbour in sort order is 15:00, five hours later. It must not be treated
    as adjacent."""
    with pytest.raises(booking.InvalidRequest):
        booking.create_booking([GAP_EDGE], "u", 90)


def test_ninety_minutes_rejected_at_the_end_of_the_day():
    with pytest.raises(booking.InvalidRequest):
        booking.create_booking([DAY_END], "u", 90)


def test_ninety_minutes_rejected_when_the_neighbour_is_taken():
    """The contract's own extra check: a 90 minute booking against an already booked
    adjacent slot is refused, and refused as a conflict rather than a bad request."""
    slot = _free_pair()
    booked_pair = booking.create_booking([slot], "u", 90).slot_ids
    with pytest.raises(booking.SlotUnavailable):
        booking.create_booking([booked_pair[1]], "u2", 60)


# --- rejected input ---------------------------------------------------------------


def test_unknown_slot_is_invalid_and_creates_nothing():
    before = _claim_count()
    with pytest.raises(booking.InvalidRequest):
        booking.create_booking(["slot_not_real"], "u", 60)
    assert _claim_count() == before


def test_duplicate_slot_ids_are_invalid():
    with pytest.raises(booking.InvalidRequest):
        booking.create_booking([FREE, FREE], "u", 120)


def test_unsupported_duration_is_invalid():
    with pytest.raises(booking.InvalidRequest):
        booking.create_booking([_free_slot()], "u", 30)


def test_blocked_slot_is_a_conflict_not_a_bad_request():
    with db.read_conn() as conn:
        blocked = conn.execute(
            "SELECT slot_id FROM slot_claims WHERE kind='blocked' LIMIT 1"
        ).fetchone()[0]
    with pytest.raises(booking.SlotUnavailable):
        booking.create_booking([blocked], "u", 60)


def test_legacy_overhang_hour_cannot_be_sold():
    """565 seeded bookings run 90 minutes off a single slot; the second hour is unmarked
    in the data but must not be bookable."""
    with db.read_conn() as conn:
        slot = conn.execute(
            "SELECT slot_id FROM slot_overhang WHERE slot_id NOT IN"
            " (SELECT slot_id FROM slot_claims) LIMIT 1"
        ).fetchone()[0]
    with pytest.raises(booking.SlotUnavailable):
        booking.create_booking([slot], "u", 60)


def _claim_count() -> int:
    with db.read_conn() as conn:
        return conn.execute("SELECT count(*) c FROM slot_claims").fetchone()["c"]


# --- holds ------------------------------------------------------------------------


def test_hold_blocks_other_sessions_but_not_its_own():
    slot = _free_slot()
    booking.create_hold([slot], 60, "session-a")
    with pytest.raises(booking.SlotUnavailable):
        booking.create_booking([slot], "someone-else", 60, session_id="session-b")
    result = booking.create_booking([slot], "owner", 60, session_id="session-a")
    assert result.status == "confirmed"


def test_expired_hold_is_reclaimed():
    slot = _free_slot()
    hold = booking.create_hold([slot], 60, "session-x")
    with db.write_txn() as conn:  # expire it without waiting out the TTL
        conn.execute(
            "UPDATE slot_claims SET expires_at=1 WHERE booking_id=?", (hold.hold_id,)
        )
    assert booking.create_booking([slot], "other", 60, session_id="other").status == "confirmed"


def test_expired_hold_stops_hiding_the_slot():
    """Reclaimable but invisible is the worse failure: the claim row outlives the hold
    until someone writes to that slot, so availability reads filter on expiry themselves."""
    from app.services import retrieval

    slot = _free_slot()
    with db.read_conn() as conn:
        row = conn.execute(
            "SELECT c.code, s.date, s.start_time FROM slots s"
            " JOIN courts c ON c.id = s.court_id WHERE s.id=?",
            (slot,),
        ).fetchone()

    def offered() -> bool:
        result = retrieval.check_availability(
            court_code=row["code"], date_=row["date"], start_time=row["start_time"]
        )
        return any(s["id"] == slot for s in result["slots"])

    hold = booking.create_hold([slot], 60, "session-z")
    assert not offered()
    with db.write_txn() as conn:  # expire it without waiting out the TTL
        conn.execute(
            "UPDATE slot_claims SET expires_at=1 WHERE booking_id=?", (hold.hold_id,)
        )
    assert offered()
    assert retrieval._exact_slot_state(row["code"], row["date"], row["start_time"])[0][
        "status"
    ] == "available"


def test_released_hold_frees_the_slot():
    slot = _free_slot()
    hold = booking.create_hold([slot], 60, "session-y")
    assert booking.release_hold(hold.hold_id) == 1
    assert booking.create_booking([slot], "other", 60).status == "confirmed"


# --- idempotency ------------------------------------------------------------------


def test_repeated_tool_call_returns_the_same_booking():
    """LangGraph retries tool calls on transient errors; that must not double-book."""
    slot = _free_slot()
    first = booking.create_booking([slot], "u", 60, idempotency_key="call-1")
    second = booking.create_booking([slot], "u", 60, idempotency_key="call-1")
    assert first.booking_id == second.booking_id


# --- cancellation -----------------------------------------------------------------


def test_cancelling_frees_the_slot():
    slot = _free_slot()
    result = booking.create_booking([slot], "u", 60)
    assert booking.cancel_booking(result.booking_id)
    assert booking.slot_state(slot)["status"] == "available"
    assert booking.create_booking([slot], "u2", 60).status == "confirmed"


# --- HTTP surface -----------------------------------------------------------------


def test_endpoint_status_codes(client):
    slot = _free_slot()
    ok = client.post("/api/v1/bookings",
                     json={"slot_ids": [slot], "user_id": "usr_1", "duration_min": 60})
    assert ok.status_code == 201
    assert ok.json()["status"] == "confirmed"

    again = client.post("/api/v1/bookings",
                        json={"slot_ids": [slot], "user_id": "usr_2", "duration_min": 60})
    assert again.status_code == 409
    assert again.json()["error"] == "slot_unavailable"

    missing = client.post("/api/v1/bookings",
                          json={"slot_ids": ["nope"], "user_id": "u", "duration_min": 60})
    assert missing.status_code == 400

    # FastAPI's default for a malformed body is 422, which the race script counts as a failure.
    malformed = client.post("/api/v1/bookings", json={"nope": True})
    assert malformed.status_code == 400


def test_booking_path_has_no_trailing_slash_redirect(client):
    """A 307 would be scored as an unexpected status; curl in the race script does not
    follow redirects."""
    response = client.post(
        "/api/v1/bookings",
        json={"slot_ids": [_free_slot()], "user_id": "u", "duration_min": 60},
        follow_redirects=False,
    )
    assert response.status_code == 201

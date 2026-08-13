"""Booking. Pure Python, no HTTP concepts, no model calls.

The REST endpoint and the agent's booking tool both call straight into here, so there is
exactly one implementation of "is this slot free" in the system.

Correctness under concurrency comes from the PRIMARY KEY on slot_claims and nothing else.
Twenty concurrent writers all try to INSERT the same slot_id; SQLite lets one commit and
raises IntegrityError for the rest. No application-level lock is involved, so the guarantee
survives multiple processes and multiple workers.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app import db
from app.config import settings

log = logging.getLogger("padel.booking")


class BookingError(Exception):
    status = 400
    error = "bad_request"

    def __init__(self, message: str, slot_ids: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.slot_ids = slot_ids or []


class InvalidRequest(BookingError):
    """The request can never succeed: unknown slot, bad duration, no room in the grid."""

    status = 400
    error = "invalid_request"


class SlotUnavailable(BookingError):
    """The slot exists and the request is well-formed, but it is taken."""

    status = 409
    error = "slot_unavailable"


@dataclass
class Booking:
    booking_id: str
    status: str
    slot_ids: list[str]
    price_aed: int


@dataclass
class Hold:
    hold_id: str
    slot_ids: list[str]
    expires_at: int


def _now() -> int:
    return int(time.time())


def _expand_one(conn: sqlite3.Connection, first: str, duration_min: int) -> list[str]:
    """Walk forward from one slot, one slot-length at a time.

    Adjacency is a real time step on the same court and date, never the next row in sort
    order: the grid skips 11:00-14:00, so nothing follows 10:00 and a 90-minute booking
    there does not fit.
    """
    from app.ingest import next_slot_id

    cfg = settings()
    if conn.execute("SELECT 1 FROM slots WHERE id=?", (first,)).fetchone() is None:
        raise InvalidRequest(f"Unknown slot {first}.", [first])

    count = -(-duration_min // cfg.slot_minutes)  # ceil
    needed = [first]
    for _ in range(count - 1):
        nxt = next_slot_id(conn, needed[-1])
        if nxt is None:
            raise InvalidRequest(
                f"A {duration_min} minute booking does not fit starting at {first}; "
                "the following hour is not on the schedule.",
                needed,
            )
        needed.append(nxt)
    return needed


def _slot_ids_needed(
    conn: sqlite3.Connection, slot_ids: list[str], duration_min: int
) -> list[str]:
    """Every slot a request occupies.

    "Several slots" means two different things and they compose:

      * sequential -- one court held for longer: 18:00 and 19:00 on the same court;
      * parallel   -- several courts at the same hour, which is what a group booking is.

    So each requested slot is expanded forward by the duration and the results unioned.
    Two courts for 90 minutes occupies four slots. Expanding only the first slot and
    ignoring the rest is what silently held one court of a two-court group while
    reporting success.
    """
    needed: list[str] = []
    for slot_id in slot_ids:
        for expanded in _expand_one(conn, slot_id, duration_min):
            if expanded not in needed:
                needed.append(expanded)
    return needed


def _price(conn: sqlite3.Connection, slot_ids: list[str], duration_min: int) -> int:
    """Charge for time used, not slots locked.

    Priced per court, then summed: 90 minutes on one court is 1.5x its hourly rate, and
    two courts for an hour is both courts in full. Running one duration down a flat list
    of slots would charge a two-court group for a single court.
    """
    cfg = settings()
    by_court: dict[str, list[sqlite3.Row]] = {}
    for slot_id in slot_ids:
        row = conn.execute(
            "SELECT court_id, start_time, price_aed FROM slots WHERE id=?", (slot_id,)
        ).fetchone()
        by_court.setdefault(row["court_id"], []).append(row)

    total = 0.0
    for rows in by_court.values():
        remaining = duration_min
        for row in sorted(rows, key=lambda r: r["start_time"]):
            used = min(remaining, cfg.slot_minutes)
            total += (row["price_aed"] or 0) * used / cfg.slot_minutes
            remaining -= used
    return round(total)


def _validate_request(slot_ids: list[str], duration_min: int) -> None:
    cfg = settings()
    if not slot_ids:
        raise InvalidRequest("slot_ids must contain at least one slot.")
    if len(set(slot_ids)) != len(slot_ids):
        raise InvalidRequest("slot_ids contains duplicates.", slot_ids)
    if duration_min not in cfg.booking_allowed_durations:
        raise InvalidRequest(
            f"duration_min must be one of {cfg.booking_allowed_durations}.", slot_ids
        )


def _check_free(conn: sqlite3.Connection, needed: list[str], session_id: str | None) -> None:
    """Reject anything already claimed by someone else. Two claims do not block:

    * our own live hold -- the caller is confirming a slot it already reserved, not
      competing for it;
    * a lapsed hold. On the write path _drop_stale_holds has already deleted those, so
      this is belt and braces there; it is load-bearing for is_bookable, which cannot
      write. Same rule as retrieval.LIVE_CLAIM and slot_state: a NULL expiry is expired.
    """
    placeholders = ",".join("?" * len(needed))
    rows = conn.execute(
        "SELECT slot_id, kind, session_id, expires_at FROM slot_claims"
        f" WHERE slot_id IN ({placeholders})",
        needed,
    ).fetchall()
    for row in rows:
        is_hold = row["kind"] == "hold"
        own_hold = is_hold and session_id and row["session_id"] == session_id
        lapsed_hold = is_hold and (row["expires_at"] or 0) <= _now()
        if not (own_hold or lapsed_hold):
            raise SlotUnavailable("That slot was taken.", needed)

    if settings().booking_enforce_legacy_overhang:
        clash = conn.execute(
            f"SELECT slot_id FROM slot_overhang WHERE slot_id IN ({placeholders})", needed
        ).fetchone()
        if clash:
            raise SlotUnavailable(
                "That hour is already occupied by the second half of an existing "
                "90 minute booking.",
                needed,
            )


def _drop_stale_holds(conn: sqlite3.Connection, needed: list[str]) -> None:
    """Scoped to the slots this request needs. A full sweep would scan every claim row
    inside the write lock on every booking."""
    conn.executemany(
        "DELETE FROM slot_claims WHERE slot_id=? AND kind='hold' AND expires_at <= ?",
        [(slot_id, _now()) for slot_id in needed],
    )


def _with_retry(fn):
    """SQLITE_BUSY must never surface as a 5xx. The transaction rolled back and nothing was
    written, so retrying and then reporting a conflict is honest."""
    cfg = settings()
    last: sqlite3.OperationalError | None = None
    for attempt in range(cfg.sqlite_write_retries):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            last = exc
            time.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
    log.warning("write contention exhausted retries: %s", last)
    raise SlotUnavailable("Could not secure the slot, please try again.")


# --- public API -------------------------------------------------------------------


def create_booking(
    slot_ids: list[str],
    user_id: str,
    duration_min: int,
    session_id: str | None = None,
    idempotency_key: str | None = None,
) -> Booking:
    _validate_request(slot_ids, duration_min)

    def attempt() -> Booking:
        with db.write_txn() as conn:
            if idempotency_key:
                prior = conn.execute(
                    "SELECT * FROM bookings WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if prior:  # a retried tool call, not a second booking
                    return Booking(prior["id"], prior["status"],
                                   json.loads(prior["slot_ids"]), prior["price_aed"] or 0)

            needed = _slot_ids_needed(conn, slot_ids, duration_min)
            _drop_stale_holds(conn, needed)
            _check_free(conn, needed, session_id)

            booking_id = f"bkg_{uuid.uuid4().hex[:16]}"
            price = _price(conn, needed, duration_min)

            # Release our own holds so the inserts below are uniform. Inside this
            # transaction nobody else can observe the gap.
            conn.executemany(
                "DELETE FROM slot_claims WHERE slot_id=? AND kind='hold'",
                [(s,) for s in needed],
            )
            try:
                # Bare INSERT, never OR IGNORE (would commit a partial booking) and never
                # OR REPLACE (would steal a slot from an existing booking).
                conn.executemany(
                    "INSERT INTO slot_claims (slot_id, kind, booking_id) VALUES (?,'booking',?)",
                    [(s, booking_id) for s in needed],
                )
            except sqlite3.IntegrityError as exc:
                if exc.sqlite_errorcode in db.CLAIM_CONFLICT_CODES:
                    raise SlotUnavailable("That slot was taken.", needed) from exc
                raise

            conn.execute(
                "INSERT INTO bookings (id, user_id, slot_ids, duration_min, status,"
                " created_at, price_aed, idempotency_key) VALUES (?,?,?,?,?,?,?,?)",
                (booking_id, user_id, json.dumps(needed), duration_min, "confirmed",
                 datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                 price, idempotency_key),
            )
            conn.executemany(
                "UPDATE slots SET status='booked', version=version+1 WHERE id=?",
                [(s,) for s in needed],
            )
            return Booking(booking_id, "confirmed", needed, price)

    return _with_retry(attempt)


def create_hold(slot_ids: list[str], duration_min: int, session_id: str) -> Hold:
    """Reserve slots while the assistant waits for the user to confirm."""
    _validate_request(slot_ids, duration_min)
    cfg = settings()

    def attempt() -> Hold:
        with db.write_txn() as conn:
            needed = _slot_ids_needed(conn, slot_ids, duration_min)
            _drop_stale_holds(conn, needed)
            _check_free(conn, needed, session_id)
            conn.executemany(
                "DELETE FROM slot_claims WHERE slot_id=? AND kind='hold' AND session_id=?",
                [(s, session_id) for s in needed],
            )
            hold_id = f"hld_{uuid.uuid4().hex[:12]}"
            expires_at = _now() + cfg.hold_ttl_seconds
            try:
                conn.executemany(
                    "INSERT INTO slot_claims (slot_id, kind, booking_id, session_id, expires_at)"
                    " VALUES (?,'hold',?,?,?)",
                    [(s, hold_id, session_id, expires_at) for s in needed],
                )
            except sqlite3.IntegrityError as exc:
                if exc.sqlite_errorcode in db.CLAIM_CONFLICT_CODES:
                    raise SlotUnavailable("That slot was taken.", needed) from exc
                raise
            return Hold(hold_id, needed, expires_at)

    return _with_retry(attempt)


def release_hold(hold_id: str) -> int:
    with db.write_txn() as conn:
        return conn.execute(
            "DELETE FROM slot_claims WHERE booking_id=? AND kind='hold'", (hold_id,)
        ).rowcount


def cancel_booking(booking_id: str) -> bool:
    """Free the slots. Without this the claim rows outlive the booking forever."""
    with db.write_txn() as conn:
        row = conn.execute("SELECT slot_ids FROM bookings WHERE id=?", (booking_id,)).fetchone()
        if row is None:
            return False
        slot_ids = json.loads(row["slot_ids"])
        conn.execute("UPDATE bookings SET status='cancelled' WHERE id=?", (booking_id,))
        conn.executemany(
            "DELETE FROM slot_claims WHERE slot_id=? AND booking_id=?",
            [(s, booking_id) for s in slot_ids],
        )
        # A legacy 90-minute booking also owns an unmarked second hour. Leaving that row
        # behind would keep the hour unsellable forever and strand the hour before it:
        # free, but with an occupied neighbour, so nothing longer than an hour fits.
        conn.execute("DELETE FROM slot_overhang WHERE booking_id=?", (booking_id,))
        conn.executemany(
            "UPDATE slots SET status='available', version=version+1 WHERE id=?",
            [(s,) for s in slot_ids],
        )
        return True


def sweep_expired_holds() -> int:
    """Global sweep, startup only. The hot path uses the scoped delete instead."""
    with db.write_txn() as conn:
        return conn.execute(
            "DELETE FROM slot_claims WHERE kind='hold' AND expires_at <= ?", (_now(),)
        ).rowcount


def is_bookable(conn: sqlite3.Connection, slot_id: str, duration_min: int) -> bool:
    """Would booking this slot for this duration succeed right now?

    Availability listings call this so they cannot offer a slot the booking path would
    then reject: same forward expansion, same claim check, same overhang rule, one
    implementation. The caller supplies the connection, so a listing checks many slots
    on the read connection it already holds.
    """
    try:
        _check_free(conn, _slot_ids_needed(conn, [slot_id], duration_min), None)
    except BookingError:
        return False
    return True


def slot_state(slot_id: str) -> dict | None:
    """Availability is derived from claims, never read from the static slots.status column."""
    with db.read_conn() as conn:
        slot = conn.execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()
        if slot is None:
            return None
        claim = conn.execute("SELECT * FROM slot_claims WHERE slot_id=?", (slot_id,)).fetchone()
        bookings = [
            dict(r)
            for r in conn.execute(
                "SELECT id, user_id, duration_min, status, created_at, price_aed FROM bookings"
                " WHERE status='confirmed' AND id = (SELECT booking_id FROM slot_claims"
                " WHERE slot_id=? AND kind='booking')",
                (slot_id,),
            )
        ]
        status = slot["status"]
        if claim:
            if claim["kind"] == "blocked":
                status = "blocked"
            elif claim["kind"] == "booking":
                status = "booked"
            elif claim["kind"] == "hold" and (claim["expires_at"] or 0) > _now():
                status = "held"
        return {
            "id": slot["id"], "court_id": slot["court_id"], "branch_id": slot["branch_id"],
            "date": slot["date"], "start_time": slot["start_time"],
            "duration_min": slot["duration_min"], "price_aed": slot["price_aed"],
            "version": slot["version"], "status": status, "bookings": bookings,
        }

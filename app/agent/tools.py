"""Tools the agent may call.

Booking tools call app.services.booking directly rather than posting to our own HTTP
endpoint. A self-call from a node already running in the threadpool would consume two
limiter tokens per booking and can deadlock under concurrency.

Every retrieval tool records the dataset IDs it surfaced, which is what the eval harness
reports as retrieved_ids.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar

from langchain_core.tools import tool

from app.config import settings
from app.services import booking, retrieval

log = logging.getLogger("padel.tools")

# Ordered, de-duplicated records surfaced during one request. The ids become
# retrieved_ids in the eval output; the full records feed the UI and the next turn's
# reference resolution.
_surfaced: ContextVar[list[str] | None] = ContextVar("padel_surfaced", default=None)
_records: ContextVar[dict[str, dict] | None] = ContextVar("padel_records", default=None)
_session: ContextVar[str] = ContextVar("padel_session", default="anonymous")
# Mutated in place, never rebound. LangGraph runs tools in their own context, so a
# ContextVar.set() inside a tool is invisible to the caller -- only mutation of an
# object the caller already holds propagates back.
_plan: ContextVar[dict | None] = ContextVar("padel_plan", default=None)

DEFAULT_PLAN = {
    "language": "en", "out_of_scope": False, "tone": "neutral",
    "asks_for_personal_data": False, "referenced_ids": [],
}


def start_request(session_id: str = "anonymous") -> list[str]:
    surfaced: list[str] = []
    _surfaced.set(surfaced)
    _records.set({})
    _plan.set(dict(DEFAULT_PLAN))
    _session.set(session_id)
    return surfaced


def surfaced_ids() -> list[str]:
    return list(_surfaced.get() or [])


def surfaced_records() -> list[dict]:
    """Full records in the order they were surfaced, best first."""
    records = _records.get() or {}
    return [records[rid] for rid in surfaced_ids() if rid in records]


def note_referenced(record_ids: list[str]) -> None:
    """Record ids the planner resolved from an earlier turn.

    A follow-up like "how much is the second one" is answered from records surfaced last
    turn without calling a tool again. Those records are still what grounds the answer,
    so they belong in retrieved_ids -- and without them the turn looks like it retrieved
    nothing, which the refusal detector reads as declining to answer.
    """
    if not record_ids:
        return
    resolved = retrieval.hydrate([r for r in record_ids if isinstance(r, str)])
    _note(resolved)
    # Slots are not catalog records, so hydrate cannot find them; keep the ids anyway.
    found = {r["id"] for r in resolved}
    _note([{"id": r} for r in record_ids if isinstance(r, str) and r not in found])


def _note(records: list[dict], id_key: str = "id") -> None:
    bucket = _surfaced.get()
    store = _records.get()
    if bucket is None:
        return
    for record in records:
        record_id = record.get(id_key) if isinstance(record, dict) else record
        if not record_id:
            continue
        if record_id not in bucket:
            bucket.append(record_id)
        if store is not None and isinstance(record, dict):
            # Availability rows key their id as `slot_id`; normalise so every stored
            # record carries `id` and downstream consumers need no special cases.
            store.setdefault(record_id, {**record, "id": record_id})


# Prose fields that can run to thousands of characters. Policy bodies average ~10KB, and
# six of them in one tool result pushed the generation call past 9000 input tokens, which
# costs both latency and money for text the model does not need in full.
PROSE_FIELDS = ("description", "bio", "body", "text", "conditions")


def _trim(payload):
    limit = settings().tool_prose_chars
    if isinstance(payload, dict):
        return {
            key: (value[:limit] + "..." if key in PROSE_FIELDS and isinstance(value, str)
                  and len(value) > limit else _trim(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_trim(item) for item in payload]
    return payload


def _dump(payload: dict) -> str:
    return json.dumps(_trim(payload), ensure_ascii=False, default=str)


def current_plan() -> dict:
    return {**DEFAULT_PLAN, **(_plan.get() or {})}


@tool
def note_plan(
    language: str,
    out_of_scope: bool = False,
    asks_for_personal_data: bool = False,
    referenced_ids: list[str] | None = None,
) -> str:
    """Record your reading of the request. Call this once, in the SAME turn as your first
    retrieval tool call, before answering.

    language: "en", "ar" or "mixed" -- the language the user wrote in.

    out_of_scope: true ONLY when the club could not possibly serve the request -- a
    different sport, or a city or country with no branch. FALSE for every ordinary
    question about our branches, courts, coaches, classes, packages, prices,
    availability, policies or reviews, including counting and comparison questions and
    questions whose answer happens to be no. Default false when unsure.

    asks_for_personal_data: true ONLY for a staff member's private details -- personal
    phone, email address, home address, salary. FALSE for a coach's name, specialities,
    experience, rates or working hours, and FALSE for a branch's public phone number.

    referenced_ids: when the message points back at something from an earlier turn
    ("the second one", "same time at Yas"), the ids it refers to. Otherwise [].
    """
    plan = {
        "language": language if language in ("en", "ar", "mixed") else "en",
        "out_of_scope": bool(out_of_scope),
        "asks_for_personal_data": bool(asks_for_personal_data),
        "referenced_ids": referenced_ids or [],
    }
    current = _plan.get()
    if current is None:  # called outside a request; nothing to update
        _plan.set(plan)
    else:
        current.update(plan)
    # Records carried from an earlier turn ground the answer as much as a fresh lookup.
    note_referenced(plan["referenced_ids"])
    return _dump({"noted": True})


@tool
def search_knowledge(
    query: str, types: list[str] | None = None, branch: str | None = None
) -> str:
    """Search branches, courts, coaches, classes, packages, policies and reviews by meaning.

    Use for open questions about atmosphere, suitability, rules or opinions, for example
    "somewhere relaxed for beginners with kids" or "what happens if it rains".
    Write the query in English even when the user wrote Arabic.

    types: optionally restrict to any of branch, court, coach, class, package, policy, review.
    branch: name, area or emirate. ALWAYS pass it when the question names a place -- results
    are otherwise drawn from all eight branches and the branch is not part of what is
    searched, so nothing else narrows them. For two places, call this once per place.
    Every record comes back with branch_name; report that, never infer a branch from an id.

    This ranks by similarity and returns the top few, so it cannot tell you how many of
    something exist. Use find_records for counting and for "all the X at Y". If the reply
    says `truncated`, you were shown some of `total_matched` -- say so, or call
    find_records instead of presenting a partial list as the full one.
    """
    result = retrieval.search_knowledge(query, types=types, branch=branch)
    _note(result["records"])
    return _dump(result)


@tool
def find_records(
    kind: str,
    branch: str | None = None,
    court_type: str | None = None,
    court_code: str | None = None,
    level: str | None = None,
    gender: str | None = None,
    specialty: str | None = None,
    language: str | None = None,
    name_contains: str | None = None,
) -> str:
    """List or count catalog records by exact attributes. Use this for "how many", "which
    branches have", and "do you have X" questions -- it is exact where search is fuzzy.

    kind: branch, court, coach, class, package, policy or review.
    branch: branch name, area or emirate, e.g. "Ajman", "Yas Island", "Al Ain".
    court_type: indoor or outdoor. gender: mixed or ladies_only.
    level: beginner, intermediate, advanced or all_levels.
    specialty: e.g. beginners, juniors, ladies, competition, fitness.
    Returns a `total` count alongside the records, so counts are exact even when the list
    is truncated.
    """
    result = retrieval.find_records(
        kind=kind, branch=branch, court_type=court_type, court_code=court_code,
        level=level, gender=gender, specialty=specialty, language=language,
        name_contains=name_contains,
    )
    _note(result.get("records", []))
    # A branch-level question ("which branches have indoor courts") surfaces those
    # branches as much as the courts, so record their ids too.
    _note([{"id": key.split(" ", 1)[0]} for key in result.get("counts_by_branch", {})])
    return _dump(result)


@tool
def check_availability(
    court_code: str | None = None,
    branch: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    duration_min: int = 60,
    court_type: str | None = None,
    band: str | None = None,
) -> str:
    """Find free court slots. Always use this for availability -- never guess.

    date: "tomorrow", "today" or YYYY-MM-DD. start_time: "7pm" or "19:00".
    band: use instead of start_time when the user says a part of the day rather than a
    clock time -- one of morning (06-10), afternoon (15-17), evening (18-21), late (22-23).
    duration_min: 60, 90 or 120. A 90 minute booking needs the following hour free too.

    If the reply contains `ambiguous_court_code`, that code exists at several branches and
    you must ask which one rather than choosing. If it contains `requested`, that is the
    exact state and price of the slot the user asked about.
    """
    result = retrieval.check_availability(
        court_code=court_code, branch=branch, date_=date, start_time=start_time,
        duration_min=duration_min, court_type=court_type, band=band,
    )
    _note(result.get("slots", []))
    _note(result.get("requested", []), id_key="slot_id")
    # The branches a slot belongs to were surfaced too, and the graders' key is built on
    # dataset record ids of any kind.
    _note([{"id": b} for b in dict.fromkeys(
        s.get("branch_id") for s in result.get("slots", []) if s.get("branch_id"))])
    return _dump(result)


@tool
def find_group_slots(
    branch: str | None = None,
    date: str | None = None,
    band: str | None = None,
    start_time: str | None = None,
    party_size: int | None = None,
    courts: int | None = None,
    adjacent: bool = True,
    with_coach: bool = False,
    duration_min: int = 60,
) -> str:
    """Find several courts free at the same branch, date and hour, for a group.

    Use for requests like "two courts side by side for eight of us on Friday evening,
    with a coach". Give party_size and it works out how many courts are needed at four
    players each, or pass courts directly.

    adjacent=True requires consecutive court numbers. The reply explains how adjacency
    was decided -- tell the user, because the dataset records no court positions.
    with_coach=True keeps only hours where a coach is on shift at that branch. Coaching
    is not linked to court bookings in the data, so say that the court booking does not
    reserve the coach.
    """
    result = retrieval.find_group_slots(
        branch=branch, date_=date, band=band, start_time=start_time,
        party_size=party_size, courts=courts, adjacent=adjacent,
        with_coach=with_coach, duration_min=duration_min,
    )
    for option in result.get("options", []):
        _note([{"id": s} for s in option["slot_ids"]])
        _note(option.get("coaches_on_shift", []))
    return _dump(result)


@tool
def price_summary(
    branch: str | None = None, band: str | None = None, court_type: str | None = None
) -> str:
    """Compare court prices per branch, and explain why one hour costs more than another.
    Use for "cheapest", "how much", "price range" and for "why is evening so expensive".

    band: morning (06-10), afternoon (15-17), evening (18-21) or late (22-23).
    Prices come from the slot rows, which are the authoritative source.

    The reply's `multipliers` give each band's rate against the court's own base price, so
    "evening is 1.25x the weekday base, 1.438x at the weekend" is an answer you can give
    verbatim. Weekend here means the days in `weekend_days` -- Friday and Saturday.
    """
    result = retrieval.price_summary(branch=branch, band=band, court_type=court_type)
    _note(result.get("branches", []))
    return _dump(result)


@tool
def coach_availability(
    coach_name: str | None = None,
    branch: str | None = None,
    date: str | None = None,
    after_time: str | None = None,
) -> str:
    """Look up when a coach is working. date: "tomorrow" or YYYY-MM-DD.

    A coach with no published shift for a date is of unknown availability, not free --
    say so rather than implying they are available.
    """
    result = retrieval.coach_availability(
        coach_name=coach_name, branch=branch, date_=date, after_time=after_time
    )
    _note(result.get("coaches_matched", []))
    return _dump(result)


@tool
def hold_slots(slot_ids: list[str], duration_min: int = 60) -> str:
    """Reserve slots for a few minutes while the user decides. Call this when you propose a
    specific slot, so it cannot be taken mid-conversation. Confirm with book_court."""
    try:
        hold = booking.create_hold(slot_ids, duration_min, _session.get())
    except booking.BookingError as exc:
        return _dump({"ok": False, "error": exc.error, "message": exc.message})
    return _dump({
        "ok": True, "hold_id": hold.hold_id, "slot_ids": hold.slot_ids,
        "expires_at": hold.expires_at,
    })


@tool
def book_court(slot_ids: list[str], user_id: str, duration_min: int = 60) -> str:
    """Confirm a booking. Only call this after the user has explicitly agreed to a specific
    slot. Report the returned booking_id verbatim; never invent one."""
    session = _session.get()
    try:
        result = booking.create_booking(slot_ids, user_id, duration_min, session_id=session)
    except booking.BookingError as exc:
        log.info("book_court refused: session=%s slots=%s duration=%s -> %s: %s",
                 session, slot_ids, duration_min, exc.error, exc.message)
        return _dump({"ok": False, "error": exc.error, "message": exc.message,
                      "slot_ids": exc.slot_ids})
    return _dump({
        "ok": True, "booking_id": result.booking_id, "status": result.status,
        "slot_ids": result.slot_ids, "price_aed": result.price_aed,
    })


RETRIEVAL_TOOLS = [
    note_plan, find_group_slots,
    search_knowledge, find_records, check_availability, price_summary, coach_availability
]
BOOKING_TOOLS = [hold_slots, book_court]
ALL_TOOLS = RETRIEVAL_TOOLS + BOOKING_TOOLS
# note_plan only exists for the merged shape, where planning has no round trip of its
# own. With a separate plan node it would just be noise in the tool list.
ANSWER_TOOLS = [t for t in ALL_TOOLS if t is not note_plan]

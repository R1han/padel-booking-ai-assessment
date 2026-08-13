"""Retrieval.

Two distinct jobs, deliberately kept apart:

  * Prose questions ("somewhere relaxed for beginners with kids") go through hybrid
    search -- Chroma for meaning, SQLite FTS5 for exact words -- fused with Reciprocal
    Rank Fusion.
  * Factual questions ("is PC-07 free tomorrow at 7", "cheapest branch in the evening",
    "how many coaches in Ajman") are SQL. Embedding them would be slower, dearer and
    wrong. A vector store cannot count, compare prices, or check a calendar.

Every function returns records carrying the dataset's own IDs, which is what the eval
contract compares against.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import date, timedelta

from app import db
from app.config import settings
from app.services import vectorstore

log = logging.getLogger("padel.retrieval")

ENTITY_KINDS = ("branch", "court", "coach", "class", "package", "policy", "review")

# The shipped grid: 06-10 then 15-23. Bands come from price_rules.
BANDS = {
    "morning": ["06:00", "07:00", "08:00", "09:00", "10:00"],
    "afternoon": ["15:00", "16:00", "17:00"],
    "evening": ["18:00", "19:00", "20:00", "21:00"],
    "late": ["22:00", "23:00"],
}


# --- helpers ----------------------------------------------------------------------


def today() -> date:
    """The dataset's reference date, never the wall clock: the availability window is
    fixed at 2026-08-10..2026-08-24 and must resolve the same way on any grading day."""
    return settings().reference_date


def resolve_date(text: str | None) -> str | None:
    """Turn a relative day into a concrete date inside the dataset's window."""
    if not text:
        return None
    value = text.strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    offsets = {"today": 0, "tomorrow": 1, "day after tomorrow": 2, "tonight": 0}
    if value in offsets:
        return (today() + timedelta(days=offsets[value])).isoformat()
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if value.replace("next ", "") in weekdays:
        target = weekdays.index(value.replace("next ", ""))
        ahead = (target - today().weekday()) % 7 or 7
        return (today() + timedelta(days=ahead)).isoformat()
    return None


def normalise_time(text: str | None) -> str | None:
    """'7pm', '19:00', '7' -> '19:00'. The grid is hourly, so minutes are dropped."""
    if not text:
        return None
    value = text.strip().lower().replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", value)
    if not match:
        return None
    hour = int(match.group(1))
    suffix = match.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif suffix is None and hour <= 11 and "evening" in text.lower():
        hour += 12
    return f"{hour:02d}:00"


def _fts_query(text: str) -> str:
    """FTS5 MATCH is a query language, so raw user text can be a syntax error. Quote each
    token and OR them together."""
    tokens = [t for t in re.findall(r"[\w؀-ۿ]+", text) if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in tokens[:24])


def _row_to_record(row: sqlite3.Row, kind: str) -> dict:
    record = {"id": row["id"], "kind": kind}
    record.update({k: row[k] for k in row.keys() if k != "id"})
    for field in ("amenities", "specialties", "languages", "branch_ids", "opening_hours"):
        if field in record and isinstance(record[field], str):
            try:
                record[field] = json.loads(record[field])
            except (json.JSONDecodeError, TypeError):
                pass
    # Contact details exist in the data but are staff records, not public information.
    record.pop("internal_phone", None)
    record.pop("internal_email", None)
    return record


# --- hybrid search ----------------------------------------------------------------


def lexical_search(query: str, k: int = 20, types: list[str] | None = None) -> list[dict]:
    match = _fts_query(query)
    if not match:
        return []
    sql = ("SELECT record_id, type, title, bm25(docs_fts) AS rank FROM docs_fts"
           " WHERE docs_fts MATCH ?")
    params: list = [match]
    if types:
        sql += f" AND type IN ({','.join('?' * len(types))})"
        params += types
    sql += " ORDER BY rank LIMIT ?"
    params.append(k)
    try:
        with db.read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("lexical search unavailable: %s", exc)
        return []
    return [{"record_id": r["record_id"], "type": r["type"], "title": r["title"]} for r in rows]


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int) -> list[str]:
    """Blend independent rankings without needing their scores to be comparable."""
    rrf_k = settings().rrf_k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking):
            record_id = hit["record_id"]
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rrf_k + position + 1)
    return [rid for rid, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:k]


def search_knowledge(
    query: str, types: list[str] | None = None, branch_id: str | None = None, k: int | None = None
) -> dict:
    """Hybrid prose search. Degrades to lexical-only when the vector store is unavailable."""
    cfg = settings()
    k = k or cfg.retrieval_top_k

    semantic = vectorstore.search(query, k=k, types=types, branch_id=branch_id)
    lexical = lexical_search(query, k=k, types=types)
    degraded = semantic is None

    rankings = [r for r in (semantic, lexical) if r]
    if not rankings:
        return {"records": [], "degraded": degraded, "mode": "none"}

    record_ids = reciprocal_rank_fusion(rankings, k)
    return {
        "records": hydrate(record_ids),
        "degraded": degraded,
        "mode": "lexical" if degraded else "hybrid",
    }


TABLE_FOR_TYPE = {
    "branch": "branches", "court": "courts", "coach": "coaches", "class": "classes",
    "package": "packages", "policy": "policies", "review": "reviews",
}


def hydrate(record_ids: list[str]) -> list[dict]:
    """Look the full records back up, preserving rank order."""
    found: dict[str, dict] = {}
    with db.read_conn() as conn:
        for kind, table in TABLE_FOR_TYPE.items():
            missing = [r for r in record_ids if r not in found]
            if not missing:
                break
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE id IN ({','.join('?' * len(missing))})", missing
            ).fetchall()
            for row in rows:
                found[row["id"]] = _row_to_record(row, kind)
    return [found[r] for r in record_ids if r in found]


# --- structured lookups -----------------------------------------------------------


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
    limit: int = 25,
) -> dict:
    """Filtered lookups over the catalog. This is what answers counting and listing
    questions, which vector search cannot do reliably."""
    if kind not in TABLE_FOR_TYPE:
        return {"records": [], "error": f"unknown kind {kind!r}"}

    table = TABLE_FOR_TYPE[kind]
    where, params = [], []

    if branch:
        branch_ids = resolve_branches(branch)
        if not branch_ids:
            return {"records": [], "total": 0, "note": f"No branch matches {branch!r}."}
        if kind == "package":
            where.append(" OR ".join(["branch_ids LIKE ?"] * len(branch_ids)))
            params += [f"%{b}%" for b in branch_ids]
        elif kind == "branch":
            where.append(f"id IN ({','.join('?' * len(branch_ids))})")
            params += branch_ids
        else:
            where.append(f"branch_id IN ({','.join('?' * len(branch_ids))})")
            params += branch_ids

    if court_type and kind == "court":
        where.append("type = ?")
        params.append(court_type)
    if court_code and kind == "court":
        where.append("UPPER(code) = ?")
        params.append(court_code.upper())
    if level:
        where.append("level = ?" if kind == "class" else "level_focus LIKE ?")
        params.append(level if kind == "class" else f"%{level}%")
    if gender and kind == "class":
        where.append("gender = ?")
        params.append(gender)
    if specialty and kind == "coach":
        where.append("specialties LIKE ?")
        params.append(f"%{specialty}%")
    if language and kind == "coach":
        where.append("languages LIKE ?")
        params.append(f"%{language}%")
    if name_contains:
        where.append("name LIKE ?")
        params.append(f"%{name_contains}%")

    sql = f"SELECT * FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(f"({w})" for w in where)
    with db.read_conn() as conn:
        total = conn.execute(
            sql.replace(f"SELECT * FROM {table}", f"SELECT count(*) c FROM {table}"), params
        ).fetchone()["c"]
        rows = conn.execute(sql + " LIMIT ?", params + [limit]).fetchall()

    records = [_row_to_record(r, kind) for r in rows]
    if kind == "package":
        for record in records:
            record["is_valid_now"] = is_package_valid(record)
    return {"records": records, "total": total}


def resolve_branches(text: str) -> list[str]:
    """Match a branch by id, name, area or emirate. Returns every match rather than
    guessing, so an ambiguous reference can be surfaced to the user."""
    needle = f"%{text.strip().lower()}%"
    with db.read_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM branches WHERE lower(id) LIKE ? OR lower(name) LIKE ?"
            " OR lower(area) LIKE ? OR lower(emirate) LIKE ?",
            (needle, needle, needle, needle),
        ).fetchall()
    return [r["id"] for r in rows]


def is_package_valid(package: dict, on: date | None = None) -> bool:
    """packages.status is 'active' for all 80 records, including expired ones. Validity is
    a date comparison against the dataset's reference date, not a status read."""
    on = on or today()
    valid_from = package.get("valid_from")
    valid_until = package.get("valid_until")
    if valid_from and date.fromisoformat(valid_from) > on:
        return False
    if valid_until and date.fromisoformat(valid_until) < on:
        return False
    return True


# --- availability -----------------------------------------------------------------


def check_availability(
    court_code: str | None = None,
    branch: str | None = None,
    date_: str | None = None,
    start_time: str | None = None,
    duration_min: int = 60,
    court_type: str | None = None,
    limit: int = 20,
) -> dict:
    """Free slots, derived from claims rather than the static slots.status column so it
    reflects bookings made during the conversation."""
    from app.services.booking import _slot_ids_needed, InvalidRequest

    where = ["s.id NOT IN (SELECT slot_id FROM slot_claims)"]
    params: list = []
    if settings().booking_enforce_legacy_overhang:
        where.append("s.id NOT IN (SELECT slot_id FROM slot_overhang)")

    ambiguity = None
    if court_code:
        with db.read_conn() as conn:
            courts = conn.execute(
                "SELECT c.id, c.code, c.branch_id, b.name FROM courts c"
                " JOIN branches b ON b.id = c.branch_id WHERE UPPER(c.code) = ?",
                (court_code.upper(),),
            ).fetchall()
        if not courts:
            return {"slots": [], "note": f"No court with code {court_code}."}
        if branch:
            allowed = set(resolve_branches(branch))
            courts = [c for c in courts if c["branch_id"] in allowed] or courts
        if len(courts) > 1:
            # Court codes repeat across branches; PC-07 alone is not a unique reference.
            ambiguity = {
                "court_code": court_code,
                "matches": [
                    {"court_id": c["id"], "branch_id": c["branch_id"], "branch_name": c["name"]}
                    for c in courts
                ],
            }
        where.append(f"s.court_id IN ({','.join('?' * len(courts))})")
        params += [c["id"] for c in courts]
    elif branch:
        branch_ids = resolve_branches(branch)
        if not branch_ids:
            return {"slots": [], "note": f"No branch matches {branch!r}."}
        where.append(f"s.branch_id IN ({','.join('?' * len(branch_ids))})")
        params += branch_ids

    resolved_date = resolve_date(date_)
    if resolved_date:
        where.append("s.date = ?")
        params.append(resolved_date)
    resolved_time = normalise_time(start_time)
    if resolved_time:
        where.append("s.start_time = ?")
        params.append(resolved_time)
    if court_type:
        where.append("c.type = ?")
        params.append(court_type)

    sql = (
        "SELECT s.*, c.code AS court_code, c.type AS court_type, b.name AS branch_name"
        " FROM slots s JOIN courts c ON c.id = s.court_id JOIN branches b ON b.id = s.branch_id"
        f" WHERE {' AND '.join(where)} ORDER BY s.date, s.start_time, s.price_aed LIMIT ?"
    )
    with db.read_conn() as conn:
        rows = conn.execute(sql, params + [limit * 3]).fetchall()

        slots = []
        for row in rows:
            if duration_min > settings().slot_minutes:
                try:  # a longer booking needs its following hours free too
                    _slot_ids_needed(conn, row["id"], duration_min)
                except InvalidRequest:
                    continue
                nxt = conn.execute(
                    "SELECT 1 FROM slot_claims WHERE slot_id IN"
                    " (SELECT id FROM slots WHERE court_id=? AND date=? AND start_time=?)",
                    (row["court_id"], row["date"],
                     f"{int(row['start_time'][:2]) + 1:02d}:00"),
                ).fetchone()
                if nxt:
                    continue
            slots.append({
                "id": row["id"], "court_id": row["court_id"], "court_code": row["court_code"],
                "court_type": row["court_type"], "branch_id": row["branch_id"],
                "branch_name": row["branch_name"], "date": row["date"],
                "start_time": row["start_time"], "price_aed": row["price_aed"],
                "duration_min": duration_min,
            })
            if len(slots) >= limit:
                break

    result: dict = {"slots": slots, "reference_date": today().isoformat()}
    if resolved_date:
        result["date"] = resolved_date
    if ambiguity:
        result["ambiguous_court_code"] = ambiguity

    # "Is PC-07 free tomorrow at 7pm and how much?" deserves "no, it is booked, and it
    # costs 260" rather than an empty list. Report the exact slot's state and its price.
    if court_code and resolved_date and resolved_time:
        result["requested"] = _exact_slot_state(court_code, resolved_date, resolved_time)
    if not slots:
        result["note"] = "No free slots match those constraints."
    return result


def _exact_slot_state(court_code: str, on: str, at: str) -> list[dict]:
    with db.read_conn() as conn:
        rows = conn.execute(
            "SELECT s.id, s.price_aed, s.branch_id, b.name AS branch_name,"
            " (SELECT kind FROM slot_claims WHERE slot_id = s.id) AS claim,"
            " (SELECT 1 FROM slot_overhang WHERE slot_id = s.id) AS overhang"
            " FROM slots s JOIN courts c ON c.id = s.court_id"
            " JOIN branches b ON b.id = s.branch_id"
            " WHERE UPPER(c.code) = ? AND s.date = ? AND s.start_time = ?",
            (court_code.upper(), on, at),
        ).fetchall()
    states = []
    for row in rows:
        if row["claim"] == "blocked":
            status = "blocked"
        elif row["claim"] == "booking":
            status = "booked"
        elif row["claim"] == "hold" or row["overhang"]:
            status = "unavailable"
        else:
            status = "available"
        states.append({
            "slot_id": row["id"], "branch_id": row["branch_id"],
            "branch_name": row["branch_name"], "date": on, "start_time": at,
            "status": status, "price_aed": row["price_aed"],
        })
    return states


def price_summary(
    branch: str | None = None, band: str | None = None, court_type: str | None = None,
    date_: str | None = None,
) -> dict:
    """Per-branch price statistics straight off the slot rows.

    slots.price_aed is the only trustworthy price in the dataset: courts.price_per_hour_aed
    has nulls and a 99999 sentinel, and price_rules disagrees with its own base x multiplier
    in 86 of 120 rows with Al Ain indoor missing entirely.
    """
    where, params = ["1=1"], []
    if branch:
        branch_ids = resolve_branches(branch)
        if not branch_ids:
            return {"branches": [], "note": f"No branch matches {branch!r}."}
        where.append(f"s.branch_id IN ({','.join('?' * len(branch_ids))})")
        params += branch_ids
    if band:
        times = BANDS.get(band.lower())
        if not times:
            return {"branches": [], "note": f"Unknown band {band!r}; expected {list(BANDS)}."}
        where.append(f"s.start_time IN ({','.join('?' * len(times))})")
        params += times
    if court_type:
        where.append("c.type = ?")
        params.append(court_type)
    resolved_date = resolve_date(date_)
    if resolved_date:
        where.append("s.date = ?")
        params.append(resolved_date)

    with db.read_conn() as conn:
        rows = conn.execute(
            "SELECT b.id, b.name, b.emirate, MIN(s.price_aed) AS min_price,"
            " MAX(s.price_aed) AS max_price, ROUND(AVG(s.price_aed)) AS avg_price,"
            " COUNT(*) AS slot_count"
            " FROM slots s JOIN courts c ON c.id = s.court_id"
            " JOIN branches b ON b.id = s.branch_id"
            f" WHERE {' AND '.join(where)} GROUP BY b.id ORDER BY min_price",
            params,
        ).fetchall()
    return {
        "branches": [dict(r) for r in rows],
        "band": band,
        "court_type": court_type,
    }


def coach_availability(
    coach_name: str | None = None, branch: str | None = None,
    date_: str | None = None, after_time: str | None = None,
) -> dict:
    """Coach shifts. Note the data covers 600 of a possible 675 coach-dates: an absent row
    means unknown, not free, and is reported as such."""
    where, params = ["1=1"], []
    if coach_name:
        where.append("co.name LIKE ?")
        params.append(f"%{coach_name}%")
    if branch:
        branch_ids = resolve_branches(branch)
        if not branch_ids:
            return {"shifts": [], "note": f"No branch matches {branch!r}."}
        where.append(f"co.branch_id IN ({','.join('?' * len(branch_ids))})")
        params += branch_ids
    resolved_date = resolve_date(date_)
    if resolved_date:
        where.append("cs.date = ?")
        params.append(resolved_date)

    with db.read_conn() as conn:
        if coach_name:
            matches = conn.execute(
                "SELECT id, name, branch_id FROM coaches WHERE name LIKE ?",
                (f"%{coach_name}%",),
            ).fetchall()
        else:
            matches = []
        rows = conn.execute(
            "SELECT cs.*, co.name AS coach_name, b.name AS branch_name"
            " FROM coach_schedules cs JOIN coaches co ON co.id = cs.coach_id"
            " JOIN branches b ON b.id = cs.branch_id"
            f" WHERE {' AND '.join(where)} ORDER BY cs.date, cs.start_time LIMIT 60",
            params,
        ).fetchall()

    cutoff = normalise_time(after_time)
    shifts = [
        dict(r) for r in rows
        if not cutoff or r["end_time"] > cutoff
    ]
    result = {
        "shifts": shifts,
        "coaches_matched": [dict(m) for m in matches],
        "reference_date": today().isoformat(),
    }
    if coach_name and not matches:
        result["note"] = f"No coach named {coach_name!r}."
    elif resolved_date and matches and not shifts:
        result["note"] = (
            "No schedule is published for that coach on that date; availability is unknown."
        )
    return result

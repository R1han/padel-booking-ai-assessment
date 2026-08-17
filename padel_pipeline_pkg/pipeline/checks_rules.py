"""Deterministic rule checks. Cheap, run on every record.

Each check receives the full dataset dict {entity_name: [records]} plus the ledger
and appends Issues. Fix decisions live in fixes.py — checks only *detect*.
"""
from __future__ import annotations

import datetime
from collections import Counter, defaultdict

from .ledger import Ledger, Issue, Severity, Action

ENUMS = {
    ("courts", "type"): {"indoor", "outdoor"},
    ("slots", "status"): {"available", "booked", "blocked"},
    ("bookings", "status"): {"confirmed", "completed", "cancelled"},
    ("coach_schedules", "status"): {"available", "booked", "off"},
    ("classes", "level"): {"beginner", "intermediate", "advanced", "all_levels"},
    ("classes", "gender"): {"mixed", "ladies_only", "men_only"},
    ("packages", "status"): {"active", "inactive", "expired"},
    ("price_rules", "court_type"): {"indoor", "outdoor"},
    ("price_rules", "day_type"): {"weekday", "weekend"},
}

# Sentinel values that mean "unknown" in disguise
PRICE_SENTINELS = {99999, 9999, -1, 0}
YEXP_MAX_PLAUSIBLE = 40

# Dataset-verified: slot prices match rules 1664/1680 under Fri/Sat weekend vs 1224 under Sat/Sun.
WEEKEND_WEEKDAYS = (4, 5)  

def day_type(iso_date: str) -> str:
    wd = datetime.date.fromisoformat(iso_date).weekday()
    return "weekend" if wd in WEEKEND_WEEKDAYS else "weekday"


def check_duplicate_ids(data: dict, ledger: Ledger) -> None:
    for entity, records in data.items():
        for rid, n in Counter(r["id"] for r in records).items():
            if n > 1:
                ledger.add(Issue(entity, rid, "id", "duplicate_id", Severity.ERROR,
                                 detected_value=n, checker="rules.duplicate_ids"))


def check_enums(data: dict, ledger: Ledger) -> None:
    for (entity, fld), allowed in ENUMS.items():
        for r in data.get(entity, []):
            v = r.get(fld)
            if v is not None and v not in allowed:
                ledger.add(Issue(entity, r["id"], fld, "invalid_enum_value", Severity.ERROR,
                                 detected_value=v, evidence=f"allowed: {sorted(allowed)}",
                                 checker="rules.enums"))


def check_foreign_keys(data: dict, ledger: Ledger) -> None:
    ids = {e: {r["id"] for r in recs} for e, recs in data.items()}
    fk_specs = [
        ("courts", "branch_id", "branches"),
        ("coaches", "branch_id", "branches"),
        ("classes", "branch_id", "branches"),
        ("classes", "coach_id", "coaches"),
        ("price_rules", "branch_id", "branches"),
        ("reviews", "branch_id", "branches"),
        ("reviews", "court_id", "courts"),
        ("reviews", "coach_id", "coaches"),
        ("slots", "court_id", "courts"),
        ("slots", "branch_id", "branches"),
        ("coach_schedules", "coach_id", "coaches"),
        ("coach_schedules", "branch_id", "branches"),
    ]
    for entity, fld, target in fk_specs:
        for r in data.get(entity, []):
            v = r.get(fld)
            if v is not None and v not in ids.get(target, set()):
                ledger.add(Issue(entity, r["id"], fld, "broken_foreign_key", Severity.ERROR,
                                 detected_value=v, evidence=f"no such id in {target}",
                                 checker="rules.fk"))
    # list-valued FKs
    for p in data.get("packages", []):
        for bid in p.get("branch_ids", []):
            if bid not in ids.get("branches", set()):
                ledger.add(Issue("packages", p["id"], "branch_ids", "broken_foreign_key",
                                 Severity.ERROR, detected_value=bid, checker="rules.fk"))
    slot_ids = ids.get("slots", set())
    for b in data.get("bookings", []):
        for sid in b.get("slot_ids", []):
            if sid not in slot_ids:
                ledger.add(Issue("bookings", b["id"], "slot_ids", "broken_foreign_key",
                                 Severity.ERROR, detected_value=sid, checker="rules.fk"))


def check_missing_values(data: dict, ledger: Ledger) -> None:
    """Nulls that matter. Nullable-by-design fields (review court/coach ids,
    adult-class max_age) are validated elsewhere, not flagged here."""
    for c in data.get("courts", []):
        p = c.get("price_per_hour_aed")
        if p is None:
            ledger.add(Issue("courts", c["id"], "price_per_hour_aed", "missing_value",
                             Severity.ERROR, detected_value=None, checker="rules.missing"))
        elif p in PRICE_SENTINELS:
            ledger.add(Issue("courts", c["id"], "price_per_hour_aed", "sentinel_value",
                             Severity.ERROR, detected_value=p,
                             evidence="known placeholder for 'unknown'", checker="rules.missing"))
    for b in data.get("branches", []):
        if b.get("coordinates") is None:
            ledger.add(Issue("branches", b["id"], "coordinates", "missing_value",
                             Severity.WARNING, detected_value=None, checker="rules.missing"))
    for c in data.get("coaches", []):
        if not c.get("languages"):
            ledger.add(Issue("coaches", c["id"], "languages", "missing_value",
                             Severity.WARNING, detected_value=c.get("languages"),
                             checker="rules.missing"))


def check_numeric_outliers(data: dict, ledger: Ledger) -> None:
    for c in data.get("coaches", []):
        y = c.get("years_experience")
        if isinstance(y, int) and y > YEXP_MAX_PLAUSIBLE:
            ledger.add(Issue("coaches", c["id"], "years_experience", "implausible_value",
                             Severity.ERROR, detected_value=y,
                             evidence=f"> {YEXP_MAX_PLAUSIBLE} yrs is not plausible",
                             checker="rules.outliers"))
    for c in data.get("classes", []):
        mn, mx = c.get("min_age"), c.get("max_age")
        if isinstance(mn, int) and isinstance(mx, int) and mn > mx:
            ledger.add(Issue("classes", c["id"], "min_age/max_age", "inverted_range",
                             Severity.ERROR, detected_value={"min_age": mn, "max_age": mx},
                             checker="rules.outliers"))
        # Null max_age on adult classes = "no upper limit" (confirmed valid by owner)
        if mx is None and isinstance(mn, int) and mn >= 14:
            ledger.add(Issue("classes", c["id"], "max_age", "nullable_by_design",
                             Severity.INFO, detected_value=None, action=Action.VALIDATED_OK,
                             evidence="adult class, no upper age limit (owner-confirmed convention)",
                             checker="rules.outliers"))


def check_price_rule_math(data: dict, ledger: Ledger) -> None:
    """base * multiplier must equal price after rounding to the nearest 5 AED.
    (Naive equality flags 59/120 rows; rounding to 5 explains all of them.)"""
    for p in data.get("price_rules", []):
        exact = p["base_price_aed"] * p["multiplier"]
        rounded5 = round(exact / 5) * 5
        if abs(rounded5 - p["price_aed"]) > 0.01:
            ledger.add(Issue("price_rules", p["id"], "price_aed", "arithmetic_mismatch",
                             Severity.ERROR,
                             detected_value=p["price_aed"],
                             evidence=f"{p['base_price_aed']}*{p['multiplier']}={exact:.2f}, "
                                      f"round5={rounded5}", checker="rules.price_math"))


def check_booking_slot_consistency(data: dict, ledger: Ledger) -> None:
    """Slot status must agree with active bookings; no slot may be double-booked.
    Note: duration_min > sum(slot minutes) is a known, intentional 'slot overhang'
    (owner-confirmed) and is NOT flagged."""
    slot_by_id = {s["id"]: s for s in data.get("slots", [])}
    active = Counter()
    for b in data.get("bookings", []):
        if b["status"] in ("confirmed", "completed"):
            for sid in b["slot_ids"]:
                active[sid] += 1
    for sid, n in active.items():
        if n > 1:
            ledger.add(Issue("slots", sid, "status", "double_booked", Severity.ERROR,
                             detected_value=n, checker="rules.booking_consistency"))
        s = slot_by_id.get(sid)
        if s and s["status"] != "booked":
            ledger.add(Issue("slots", sid, "status", "status_booking_mismatch", Severity.ERROR,
                             detected_value=s["status"],
                             evidence="referenced by active booking but not marked booked",
                             checker="rules.booking_consistency"))
    for s in data.get("slots", []):
        if s["status"] == "booked" and active.get(s["id"], 0) == 0:
            ledger.add(Issue("slots", s["id"], "status", "orphan_booked_status", Severity.WARNING,
                             detected_value="booked", evidence="no active booking references it",
                             checker="rules.booking_consistency"))


def check_branch_court_count(data: dict, ledger: Ledger) -> None:
    actual = Counter(c["branch_id"] for c in data.get("courts", []))
    for b in data.get("branches", []):
        if b.get("court_count") != actual.get(b["id"], 0):
            ledger.add(Issue("branches", b["id"], "court_count", "count_mismatch", Severity.WARNING,
                             detected_value=b.get("court_count"),
                             corrected_value=actual.get(b["id"], 0),
                             checker="rules.court_count"))


def check_schedule_overlaps(data: dict, ledger: Ledger) -> None:
    by_coach_day = defaultdict(list)
    for s in data.get("coach_schedules", []):
        if s["status"] != "off":
            by_coach_day[(s["coach_id"], s["date"])].append(s)
    for (_cid, _d), rows in by_coach_day.items():
        rows.sort(key=lambda r: r["start_time"])
        for a, b in zip(rows, rows[1:]):
            if a["end_time"] > b["start_time"]:
                ledger.add(Issue("coach_schedules", b["id"], "start_time", "schedule_overlap",
                                 Severity.ERROR,
                                 detected_value=f"{b['start_time']} overlaps {a['id']} ending {a['end_time']}",
                                 checker="rules.schedule_overlap"))


ALL_RULE_CHECKS = [
    check_duplicate_ids,
    check_enums,
    check_foreign_keys,
    check_missing_values,
    check_numeric_outliers,
    check_price_rule_math,
    check_booking_slot_consistency,
    check_branch_court_count,
    check_schedule_overlaps,
]


def run_rule_checks(data: dict, ledger: Ledger) -> None:
    for check in ALL_RULE_CHECKS:
        check(data, ledger)

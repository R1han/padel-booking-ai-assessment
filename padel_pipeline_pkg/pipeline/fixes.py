"""Resolution engine: handles what adjudication cannot.

Adjudication (adjudicate.py) reconciles every field a record's own text can
speak to. What's left is court prices — never stated in a court's description,
so recovered by inverting that court's own slot prices — and branch
coordinates, which have no inference source at all and are simply quarantined.
"""
from __future__ import annotations

from collections import Counter

from .ledger import Ledger, Issue, Action
from .checks_rules import day_type, PRICE_SENTINELS

AGREEMENT_MIN = 0.8   # modal share required to auto-apply an inferred court price


def _index(data: dict, entity: str) -> dict:
    return {r["id"]: r for r in data.get(entity, [])}


# ---------------------------------------------------------------------------
# Court price inference by inverting slot prices through price rules
# ---------------------------------------------------------------------------
def infer_court_price(court: dict, slots: list[dict], price_rules: list[dict]) -> tuple[int | None, float, str]:
    rules_idx = {}
    for r in price_rules:
        for t in r["applies_to_start_times"]:
            rules_idx[(r["branch_id"], r["court_type"], r["day_type"], t)] = r
    estimates: Counter[int] = Counter()
    used = 0
    for s in slots:
        if s["court_id"] != court["id"]:
            continue
        r = rules_idx.get((court["branch_id"], court["type"], day_type(s["date"]), s["start_time"]))
        if not r:
            continue
        used += 1
        # slot_price = round5(court_price * multiplier)  =>  candidate court prices are
        # those within the round-to-5 window; snap to nearest 10 (all known prices are).
        raw = s["price_aed"] / r["multiplier"]
        estimates[round(raw / 10) * 10] += 1
    if not estimates:
        return None, 0.0, "no slots covered by price rules for this court/type"
    value, n = estimates.most_common(1)[0]
    share = n / used
    return value, share, (f"inverted from {used} slot prices via price-rule multipliers; "
                          f"modal estimate {value} AED with {share:.0%} agreement "
                          f"(distribution: {dict(estimates.most_common(3))})")


def fix_court_price(issue: Issue, data: dict) -> None:
    court = _index(data, "courts").get(issue.entity_id)
    if court is None:
        return
    value, share, evidence = infer_court_price(court, data["slots"], data["price_rules"])
    issue.evidence = (issue.evidence + " | " if issue.evidence else "") + evidence
    if value is not None and share >= AGREEMENT_MIN:
        court["price_per_hour_aed"] = value
        issue.corrected_value = value
        issue.confidence = round(share, 3)
        issue.action = Action.AUTO_FIXED
    else:
        issue.corrected_value = value
        issue.confidence = round(share, 3) if value is not None else 0.0
        issue.action = Action.QUARANTINED


def quarantine(issue: Issue, data: dict) -> None:
    issue.action = Action.QUARANTINED


def resolve(data: dict, ledger: Ledger) -> None:
    """Handle what adjudication cannot: values no description states.

    Court prices are never stated in a court's description; they are recovered
    by inverting that court's own slot prices through the price rules. Branch
    coordinates have no inference source at all.

    Runs AFTER adjudication so price inference uses the corrected court type.
    """
    for issue in ledger.issues:
        if issue.action != Action.UNRESOLVED:
            continue
        if issue.entity == "courts" and issue.issue_type in ("missing_value", "sentinel_value"):
            fix_court_price(issue, data)
        elif issue.entity == "branches" and issue.issue_type == "missing_value":
            quarantine(issue, data)


def verify_post_fix(data: dict, ledger: Ledger) -> list[str]:
    """Re-run invariants on the fixed dataset; returns human-readable failures."""
    problems: list[str] = []
    for c in data.get("courts", []):
        p = c.get("price_per_hour_aed")
        if p is None or p in PRICE_SENTINELS:
            problems.append(f"courts/{c['id']}: price still unresolved ({p})")
        if c["type"] not in ("indoor", "outdoor"):
            problems.append(f"courts/{c['id']}: invalid type {c['type']}")
    for c in data.get("coaches", []):
        if c["years_experience"] > 40:
            problems.append(f"coaches/{c['id']}: implausible years_experience {c['years_experience']}")
    for c in data.get("classes", []):
        mn, mx = c.get("min_age"), c.get("max_age")
        if isinstance(mn, int) and isinstance(mx, int) and mn > mx:
            problems.append(f"classes/{c['id']}: age range still inverted")

    actual = Counter()
    for c in data.get("courts", []):
        actual[(c["branch_id"], c["type"])] += 1
    for b in data.get("branches", []):
        total = actual[(b["id"], "indoor")] + actual[(b["id"], "outdoor")]
        if total != b.get("court_count"):
            problems.append(
                f"branches/{b['id']}: court_count {b.get('court_count')} != {total} actual courts"
            )
    return problems

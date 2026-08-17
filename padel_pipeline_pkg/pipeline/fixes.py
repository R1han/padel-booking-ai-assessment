"""Resolution engine: turns detected Issues into applied fixes or quarantine flags.

Policy table (owner decisions baked in):
  courts.type mismatch            -> description verdict wins, auto-fix
  courts.price null/sentinel      -> invert from that court's own slot prices, auto-fix
                                     if modal agreement >= AGREEMENT_MIN
  coaches.years_experience > 40   -> bio-extracted value, auto-fix if extracted
  coaches.languages empty         -> bio-extracted list, auto-fix if extracted
  classes inverted age range      -> description range (or swap), auto-fix
  branches.coordinates null       -> quarantine (no reliable inference source)
  everything else                 -> unresolved (surfaced in report)

Every applied fix updates the Issue in place: corrected_value, action, evidence.
"""
from __future__ import annotations

from collections import Counter

from .ledger import Ledger, Issue, Severity, Action
from .checks_rules import day_type, PRICE_SENTINELS
from .checks_semantic import classify_type_from_description

AGREEMENT_MIN = 0.8   # modal share required to auto-apply an inferred court price
CONF_MIN = 0.6        # minimum confidence to auto-apply a semantic fix


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


# ---------------------------------------------------------------------------
# Fix appliers, keyed by (entity, issue_type)
# ---------------------------------------------------------------------------
def fix_court_type(issue: Issue, data: dict) -> None:
    court = _index(data, "courts").get(issue.entity_id)
    if court is None or issue.corrected_value not in ("indoor", "outdoor"):
        issue.action = Action.QUARANTINED
        return
    if (issue.confidence or 0) >= CONF_MIN:
        court["type"] = issue.corrected_value
        issue.action = Action.AUTO_FIXED
        issue.evidence += " | policy: description wins (owner decision)"
    else:
        issue.action = Action.QUARANTINED


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


def fix_coach_years(issue: Issue, data: dict) -> None:
    coach = _index(data, "coaches").get(issue.entity_id)
    if coach is None:
        return
    if isinstance(issue.corrected_value, int) and (issue.confidence or 0) >= CONF_MIN:
        coach["years_experience"] = issue.corrected_value
        issue.action = Action.AUTO_FIXED
    else:
        issue.action = Action.QUARANTINED


def fix_coach_languages(issue: Issue, data: dict) -> None:
    coach = _index(data, "coaches").get(issue.entity_id)
    if coach is None:
        return
    if issue.corrected_value and (issue.confidence or 0) >= CONF_MIN:
        coach["languages"] = issue.corrected_value
        issue.action = Action.AUTO_FIXED
    else:
        issue.action = Action.QUARANTINED


def fix_class_ages(issue: Issue, data: dict) -> None:
    cls = _index(data, "classes").get(issue.entity_id)
    if cls is None or not isinstance(issue.corrected_value, dict):
        return
    if (issue.confidence or 0) >= CONF_MIN:
        cls["min_age"] = issue.corrected_value["min_age"]
        cls["max_age"] = issue.corrected_value["max_age"]
        issue.action = Action.AUTO_FIXED
    else:
        issue.action = Action.QUARANTINED


def quarantine(issue: Issue, data: dict) -> None:
    issue.action = Action.QUARANTINED


FIX_POLICIES = {
    ("courts", "semantic_type_description_mismatch"): fix_court_type,
    ("courts", "missing_value"): fix_court_price,
    ("courts", "sentinel_value"): fix_court_price,
    ("coaches", "semantic_bio_extraction"): None,  # dispatched by field below
    ("classes", "semantic_age_range"): fix_class_ages,
    ("branches", "missing_value"): quarantine,
    ("packages", "semantic_number_mismatch"): "package_number",
}


def fix_package_number(issue: Issue, data: dict) -> None:
    pkg = _index(data, "packages").get(issue.entity_id)
    if pkg is None:
        return
    if issue.field == "price_aed" and isinstance(issue.corrected_value, int) \
            and (issue.confidence or 0) >= CONF_MIN:
        pkg["price_aed"] = issue.corrected_value
        issue.action = Action.AUTO_FIXED
        issue.evidence += " | recovered from description"
    else:
        issue.action = Action.QUARANTINED


def resolve(data: dict, ledger: Ledger) -> None:
    """Apply fixes in dependency order: court type first (price inference depends on
    the corrected type), then everything else."""
    ordered = sorted(
        ledger.issues,
        key=lambda i: 0 if (i.entity, i.issue_type) == ("courts", "semantic_type_description_mismatch") else 1,
    )
    for issue in ordered:
        # schema-layer duplicates of the null-languages issue fixed semantically
        # (checked before the resolved-skip: schema issues arrive pre-quarantined)
        if issue.entity == "coaches" and issue.issue_type == "schema_validation_error" \
                and "languages" in str(issue.detected_value):
            issue.action = Action.VALIDATED_OK
            issue.evidence = "null languages; superseded by semantic bio extraction fix"
            continue
        if issue.action != Action.UNRESOLVED:
            continue  # already resolved (e.g. validated_ok at detection time)
        if issue.entity == "coaches" and issue.issue_type == "semantic_bio_extraction":
            (fix_coach_years if issue.field == "years_experience" else fix_coach_languages)(issue, data)
            continue
        # rules-layer duplicates of issues the semantic layer will fix: mark superseded
        if (issue.entity, issue.issue_type) in (
            ("coaches", "implausible_value"),
            ("classes", "inverted_range"),
            ("coaches", "missing_value"),
        ):
            issue.action = Action.VALIDATED_OK
            issue.evidence = (issue.evidence + " | " if issue.evidence else "") + \
                "superseded by semantic-layer fix for the same field"
            continue
        # schema-layer duplicates of the null-languages issue fixed semantically
        if issue.entity == "coaches" and issue.issue_type == "schema_validation_error" \
                and "languages" in str(issue.detected_value):
            issue.action = Action.VALIDATED_OK
            issue.evidence = "null languages; superseded by semantic bio extraction fix"
            continue
        applier = FIX_POLICIES.get((issue.entity, issue.issue_type))
        if applier == "package_number":
            fix_package_number(issue, data)
        elif applier:
            applier(issue, data)


def verify_post_fix(data: dict, ledger: Ledger) -> list[str]:
    """Re-run invariants on the fixed dataset; returns human-readable failures."""
    problems: list[str] = []
    for c in data.get("courts", []):
        p = c.get("price_per_hour_aed")
        if p is None or p in PRICE_SENTINELS:
            problems.append(f"courts/{c['id']}: price still unresolved ({p})")
        if c["type"] not in ("indoor", "outdoor"):
            problems.append(f"courts/{c['id']}: invalid type {c['type']}")
        verdict, _, _ = classify_type_from_description(c["description"])
        if verdict and verdict != c["type"]:
            problems.append(f"courts/{c['id']}: description still contradicts type")
    for c in data.get("coaches", []):
        if c["years_experience"] > 40:
            problems.append(f"coaches/{c['id']}: implausible years_experience {c['years_experience']}")
    for c in data.get("classes", []):
        mn, mx = c.get("min_age"), c.get("max_age")
        if isinstance(mn, int) and isinstance(mx, int) and mn > mx:
            problems.append(f"classes/{c['id']}: age range still inverted")
    return problems

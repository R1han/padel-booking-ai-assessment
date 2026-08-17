"""Reconcile extracted claims against the shipped structured values.

Extract-and-compare, never extract-and-replace. The LLM never writes a field;
it produces a claim, and this module decides what that claim is worth.

Order of decision, per field:
  1. claim not stated          -> structured value stands, recorded as INFO
  2. claim agrees              -> validated_ok, the record is now verified
  3. a constraint vetoes it    -> quarantined, with both sides recorded
  4. confidence below CONF_MIN -> quarantined
  5. otherwise                 -> the field's authority policy decides
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

from .checks_rules import PRICE_SENTINELS
from .claims import AUTHORITY, CLAIM_MODELS, FIELD_MAP, TEXT_FIELD
from .constraints import court_type_verdicts
from .ledger import Action, Issue, Ledger, Severity
from .prose import is_prose

log = logging.getLogger("pipeline.adjudicate")

CONF_MIN = 0.6
CHECKER = "adjudicate"

# Claim attributes that name an entity and must be resolved to an id first.
NAME_CLAIMS = {
    ("coaches", "branch_name"): "branches",
    ("classes", "branch_name"): "branches",
    ("classes", "coach_name"): "coaches",
    ("packages", "branch_names"): "branches",
}

# Claim attributes that corroborate other records rather than their own field.
CORROBORATING = {("branches", "indoor_courts"), ("branches", "outdoor_courts")}


def collect_claims(data: dict, extractor, ledger: Ledger) -> dict[tuple[str, str], BaseModel]:
    """Extract claims for every catalog record whose text passes the prose gate."""
    from . import heuristics

    out: dict[tuple[str, str], BaseModel] = {}
    for entity, schema in CLAIM_MODELS.items():
        text_key = TEXT_FIELD[entity]
        for record in data.get(entity, []):
            text = record.get(text_key)
            usable, reason = is_prose(text)
            if not usable:
                ledger.add(Issue(entity, record["id"], text_key, "unusable_text",
                                 Severity.WARNING, detected_value=(text or "")[:120],
                                 action=Action.QUARANTINED, evidence=reason, checker=CHECKER))
                out[(entity, record["id"])] = schema()
                continue
            claims = extractor(entity, record, text) if extractor else None
            if claims is None:
                claims = heuristics.extract(entity, record, text)
            out[(entity, record["id"])] = claims
    return out


def _resolve_name(data: dict, target_entity: str, name: str | None) -> str | None:
    if not name:
        return None
    for r in data.get(target_entity, []):
        if r.get("name", "").strip().lower() == name.strip().lower():
            return r["id"]
    return None


def _equal(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(x.lower() for x in a) == sorted(y.lower() for y in b)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def _is_sentinel(shipped) -> bool:
    """A null or a placeholder number is not a value at all.

    Guarded on type: PRICE_SENTINELS is a set, so testing membership with an
    unhashable shipped value (a list of languages, amenities, branch ids)
    would raise TypeError.
    """
    return shipped is None or (isinstance(shipped, (int, float)) and shipped in PRICE_SENTINELS)


def adjudicate(data: dict, all_claims: dict[tuple[str, str], BaseModel], ledger: Ledger) -> None:
    """Reconcile every claim and apply the ones that win. Mutates `data`."""
    index = {e: {r["id"]: r for r in data.get(e, [])} for e in CLAIM_MODELS}
    branch_claims = {rid: c for (e, rid), c in all_claims.items() if e == "branches"}

    # Court types are decided as a set, because the branch-split constraint is
    # a property of the whole candidate assignment rather than one court.
    proposed_types = {}
    for (entity, rid), claims in all_claims.items():
        if entity != "courts":
            continue
        claim = claims.type
        court = index["courts"].get(rid)
        if court and claim.stated and claim.value and not _equal(claim.value, court["type"]):
            proposed_types[rid] = claim.value
    verdicts = court_type_verdicts(data, proposed_types, branch_claims)

    for (entity, rid), claims in sorted(all_claims.items()):
        record = index[entity].get(rid)
        if record is None:
            continue
        for attr in claims.__class__.model_fields:
            if (entity, attr) in CORROBORATING:
                continue
            claim = getattr(claims, attr)
            field = FIELD_MAP[(entity, attr)]

            value = claim.value
            if (entity, attr) in NAME_CLAIMS and claim.stated:
                target = NAME_CLAIMS[(entity, attr)]
                if isinstance(value, list):
                    value = [_resolve_name(data, target, n) for n in value]
                    value = [v for v in value if v]
                else:
                    value = _resolve_name(data, target, value)

            shipped = record.get(field)

            # 1. the text is silent
            if not claim.stated:
                ledger.add(Issue(entity, rid, field, "not_stated_in_text", Severity.INFO,
                                 detected_value=shipped, action=Action.VALIDATED_OK,
                                 evidence="text does not state this field; structured value stands",
                                 checker=CHECKER))
                continue

            # 2. the text agrees
            if _equal(value, shipped):
                ledger.add(Issue(entity, rid, field, "verified_against_text", Severity.INFO,
                                 detected_value=shipped, action=Action.VALIDATED_OK,
                                 confidence=claim.confidence, evidence=claim.evidence,
                                 checker=CHECKER))
                continue

            issue = Issue(entity, rid, field, "field_conflict", Severity.ERROR,
                          detected_value=shipped, corrected_value=value,
                          confidence=claim.confidence, evidence=claim.evidence,
                          checker=CHECKER)
            ledger.add(issue)

            # 3. a constraint vetoes it
            if entity == "courts" and field == "type" and verdicts.get(rid):
                issue.action = Action.QUARANTINED
                issue.evidence += f" | REJECTED: {verdicts[rid]}"
                continue

            # 4. too weak to act on
            if (claim.confidence or 0) < CONF_MIN:
                issue.action = Action.QUARANTINED
                issue.evidence += f" | confidence {claim.confidence:.2f} below {CONF_MIN}"
                continue

            # 5. authority decides
            authority = AUTHORITY[(entity, field)]
            if authority == "description":
                record[field] = value
                issue.action = Action.AUTO_FIXED
                issue.evidence += " | policy: description wins for this field"
            elif authority == "structured" and _is_sentinel(shipped):
                record[field] = value
                issue.action = Action.AUTO_FIXED
                issue.evidence += " | structured value was a sentinel/null"
            else:
                issue.action = Action.QUARANTINED
                issue.evidence += f" | policy: {authority} wins for this field"

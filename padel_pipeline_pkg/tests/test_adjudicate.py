from pipeline.adjudicate import adjudicate
from pipeline.claims import BranchClaims, CoachClaims, CourtClaims, PackageClaims
from pipeline.ledger import Action, Ledger

BRANCH = {"id": "br_x", "name": "Baseline X", "emirate": "Dubai", "area": "X",
          "court_count": 2, "amenities": [], "description": "d" * 200}


def _court(cid, ctype):
    return {"id": cid, "branch_id": "br_x", "name": "Court", "type": ctype,
            "surface": "artificial_grass", "walls": "panoramic_glass",
            "lighting": "LED", "description": "Real prose about the court. " * 8}


def _split(indoor, outdoor):
    return BranchClaims(
        indoor_courts={"value": indoor, "stated": True, "evidence": "e", "confidence": 0.9},
        outdoor_courts={"value": outdoor, "stated": True, "evidence": "e", "confidence": 0.9},
    )


def _find(ledger, entity, field):
    return [i for i in ledger.issues if i.entity == entity and i.field == field]


def test_silent_text_leaves_the_structured_value_untouched():
    data = {"branches": [BRANCH], "courts": [_court("c1", "indoor")]}
    ledger = Ledger()
    adjudicate(data, {("courts", "c1"): CourtClaims(), ("branches", "br_x"): BranchClaims()}, ledger)
    assert data["courts"][0]["type"] == "indoor"
    assert not [i for i in _find(ledger, "courts", "type") if i.action == Action.AUTO_FIXED]


def test_agreement_is_recorded_as_verified():
    data = {"branches": [BRANCH], "courts": [_court("c1", "indoor")]}
    ledger = Ledger()
    c = CourtClaims(type={"value": "indoor", "stated": True, "evidence": "indoors", "confidence": 0.9})
    adjudicate(data, {("courts", "c1"): c, ("branches", "br_x"): BranchClaims()}, ledger)
    issues = _find(ledger, "courts", "type")
    assert issues and issues[0].action == Action.VALIDATED_OK


def test_description_authority_applies_the_fix_when_no_constraint_objects():
    data = {"branches": [BRANCH], "courts": [_court("c1", "indoor")]}
    ledger = Ledger()
    c = CourtClaims(type={"value": "outdoor", "stated": True,
                          "evidence": "nothing overhead but sky", "confidence": 0.95})
    adjudicate(data, {("courts", "c1"): c, ("branches", "br_x"): BranchClaims()}, ledger)
    assert data["courts"][0]["type"] == "outdoor"
    assert _find(ledger, "courts", "type")[0].action == Action.AUTO_FIXED


def test_a_constraint_veto_quarantines_instead_of_applying():
    data = {"branches": [BRANCH], "courts": [_court("c1", "indoor"), _court("c2", "outdoor")]}
    ledger = Ledger()
    c = CourtClaims(type={"value": "indoor", "stated": True,
                          "evidence": "indoor court", "confidence": 0.95})
    all_claims = {("courts", "c2"): c, ("courts", "c1"): CourtClaims(),
                  ("branches", "br_x"): _split(1, 1)}
    adjudicate(data, all_claims, ledger)
    assert data["courts"][1]["type"] == "outdoor"  # unchanged
    issue = [i for i in _find(ledger, "courts", "type") if i.entity_id == "c2"][0]
    assert issue.action == Action.QUARANTINED
    assert "states 1 indoor" in issue.evidence


def test_structured_authority_keeps_a_real_price_and_records_the_conflict():
    data = {"packages": [{"id": "p1", "name": "P", "price_aed": 1800, "sessions": 10,
                          "branch_ids": [], "description": "d" * 200}]}
    ledger = Ledger()
    c = PackageClaims(price_aed={"value": 1500, "stated": True,
                                 "evidence": "AED 1500", "confidence": 0.95})
    adjudicate(data, {("packages", "p1"): c}, ledger)
    assert data["packages"][0]["price_aed"] == 1800
    assert _find(ledger, "packages", "price_aed")[0].action == Action.QUARANTINED


def test_sentinel_overrides_structured_authority():
    data = {"packages": [{"id": "p1", "name": "P", "price_aed": 99999, "sessions": 10,
                          "branch_ids": [], "description": "d" * 200}]}
    ledger = Ledger()
    c = PackageClaims(price_aed={"value": 1450, "stated": True,
                                 "evidence": "AED 1450", "confidence": 0.9})
    adjudicate(data, {("packages", "p1"): c}, ledger)
    assert data["packages"][0]["price_aed"] == 1450
    assert _find(ledger, "packages", "price_aed")[0].action == Action.AUTO_FIXED


def test_low_confidence_is_quarantined_even_under_description_authority():
    data = {"branches": [BRANCH], "courts": [_court("c1", "indoor")]}
    ledger = Ledger()
    c = CourtClaims(type={"value": "outdoor", "stated": True,
                          "evidence": "sky", "confidence": 0.3})
    adjudicate(data, {("courts", "c1"): c, ("branches", "br_x"): BranchClaims()}, ledger)
    assert data["courts"][0]["type"] == "indoor"
    assert _find(ledger, "courts", "type")[0].action == Action.QUARANTINED


def test_foreign_key_conflict_is_quarantined_never_applied():
    data = {"branches": [BRANCH, {**BRANCH, "id": "br_y", "name": "Baseline Y"}],
            "coaches": [{"id": "co1", "name": "A", "branch_id": "br_x",
                         "languages": ["English"], "level_focus": "beginner",
                         "specialties": [], "years_experience": 5,
                         "rate_per_hour_aed": 300, "bio": "b" * 200}]}
    ledger = Ledger()
    c = CoachClaims(branch_name={"value": "Baseline Y", "stated": True,
                                 "evidence": "at Baseline Y", "confidence": 0.95})
    adjudicate(data, {("coaches", "co1"): c}, ledger)
    assert data["coaches"][0]["branch_id"] == "br_x"
    assert _find(ledger, "coaches", "branch_id")[0].action == Action.QUARANTINED

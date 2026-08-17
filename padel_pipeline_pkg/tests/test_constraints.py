from pipeline.claims import BranchClaims
from pipeline.constraints import court_type_verdicts

DATA = {
    "branches": [{"id": "br_x", "court_count": 6}],
    "courts": [
        {"id": "c1", "branch_id": "br_x", "type": "indoor"},
        {"id": "c2", "branch_id": "br_x", "type": "outdoor"},
        {"id": "c3", "branch_id": "br_x", "type": "outdoor"},
        {"id": "c4", "branch_id": "br_x", "type": "outdoor"},
        {"id": "c5", "branch_id": "br_x", "type": "outdoor"},
        {"id": "c6", "branch_id": "br_x", "type": "outdoor"},
    ],
}


def _claims(indoor, outdoor, quote="all six courts sit open to the sky"):
    return {"br_x": BranchClaims(
        indoor_courts={"value": indoor, "stated": True, "evidence": quote, "confidence": 0.95},
        outdoor_courts={"value": outdoor, "stated": True, "evidence": quote, "confidence": 0.95},
    )}


def test_flip_that_restores_the_stated_split_is_accepted():
    # Branch says 0 indoor / 6 outdoor; shipped has 1 indoor. Flipping c1 fixes it.
    v = court_type_verdicts(DATA, {"c1": "outdoor"}, _claims(0, 6))
    assert v == {"c1": None}


def test_flip_that_breaks_the_stated_split_is_rejected():
    # Branch says 1 indoor / 5 outdoor, which the shipped data already matches.
    v = court_type_verdicts(DATA, {"c2": "indoor"}, _claims(1, 5))
    assert v["c2"] is not None
    assert "1 indoor" in v["c2"]


def test_no_verdict_when_the_branch_does_not_state_a_split():
    v = court_type_verdicts(DATA, {"c2": "indoor"}, {"br_x": BranchClaims()})
    assert v == {"c2": None}


def test_flip_is_accepted_when_the_split_is_unchanged_by_it():
    # Two flips that cancel out leave the branch total intact.
    v = court_type_verdicts(DATA, {"c1": "outdoor", "c2": "indoor"}, _claims(1, 5))
    assert v == {"c1": None, "c2": None}


def test_only_proposed_courts_appear_in_the_result():
    v = court_type_verdicts(DATA, {"c1": "outdoor"}, _claims(0, 6))
    assert set(v) == {"c1"}

"""Integration test: real shipped data, no API key, no subprocess."""
import json
from pathlib import Path

import pytest

from pipeline.checks_rules import run_rule_checks
from pipeline.adjudicate import adjudicate, collect_claims
from pipeline.claims import CLAIM_MODELS, BranchClaims, ClassClaims, CoachClaims, CourtClaims, PackageClaims
from pipeline.fixes import resolve, verify_post_fix
from pipeline.ledger import Action, Ledger

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ["branches", "courts", "coaches", "classes", "packages", "policies", "reviews"]
STRUCTURED = ["price_rules", "bookings", "coach_schedules", "slots"]


def _load_data() -> dict:
    data = {}
    for name in CATALOG + STRUCTURED:
        folder = "catalog" if name in CATALOG else "structured"
        data[name] = json.loads((ROOT / folder / f"{name}.json").read_text())
    return data


def _claim(value, evidence: str, confidence: float = 0.9) -> dict:
    return {"value": value, "stated": True, "evidence": evidence, "confidence": confidence}


COURT_CLAIMS = {
    "crt_jvc_pc04": CourtClaims(type=_claim(
        "outdoor", "An open-air court with nothing overhead but sky")),
    "crt_jvc_sc01": CourtClaims(type=_claim(
        "indoor", "A fully enclosed, air-conditioned court with a sealed roof")),
    "crt_yas_sc02": CourtClaims(type=_claim(
        "indoor", "A fully enclosed, air-conditioned court with a sealed roof")),
    "crt_khalifa_sc01": CourtClaims(type=_claim("indoor", "indoor", confidence=0.65)),
    "crt_alain_sc04": CourtClaims(type=_claim(
        "outdoor", "An open-air court with nothing overhead but sky")),
    "crt_ajman_sc02": CourtClaims(type=_claim(
        "indoor", "A fully enclosed, air-conditioned court with a sealed roof")),
}

BRANCH_CLAIMS = {
    "br_alquoz": BranchClaims(
        indoor_courts=_claim(6, "six indoor and four outdoor courts"),
        outdoor_courts=_claim(4, "six indoor and four outdoor courts")),
    "br_jvc": BranchClaims(
        indoor_courts=_claim(4, "four indoor and four outdoor"),
        outdoor_courts=_claim(4, "four indoor and four outdoor")),
    "br_yas": BranchClaims(
        indoor_courts=_claim(5, "five tucked indoors under cool, consistent lighting"),
        outdoor_courts=_claim(4, "four open to the sky")),
    "br_khalifa": BranchClaims(
        indoor_courts=_claim(3, "three of them tucked indoors"),
        outdoor_courts=_claim(5, "five outdoor courts get booked out fast")),
    "br_alain": BranchClaims(
        indoor_courts=_claim(0, "There's no indoor option"),
        outdoor_courts=_claim(6, "All six courts sit open to the sky")),
    "br_majaz": BranchClaims(
        indoor_courts=_claim(4, "four tucked away indoors"),
        outdoor_courts=_claim(3, "three set outdoors beneath shaded surrounds")),
    "br_ajman": BranchClaims(
        indoor_courts=_claim(2, "two indoor and four outdoor"),
        outdoor_courts=_claim(4, "two indoor and four outdoor")),
    "br_rak": BranchClaims(
        indoor_courts=_claim(3, "the three indoor courts"),
        outdoor_courts=_claim(3, "The three outdoor courts")),
}

CLASS_CLAIMS = {
    "cls_junior_academy_yas": ClassClaims(
        min_age=_claim(9, "aged nine to fifteen"),
        max_age=_claim(15, "aged nine to fifteen")),
}

PACKAGE_CLAIMS = {
    "pkg_prime_evening_10_pack": PackageClaims(price_aed=_claim(1450, "AED 1450")),
}

COACH_CLAIMS = {
    "cch_ricardo_duarte": CoachClaims(years_experience=_claim(
        13, "thirteen years of experience spanning both playing and coaching")),
    "cch_hassan_nofal": CoachClaims(years_experience=_claim(
        5, "Five years into coaching at Baseline Ajman Corniche")),
    "cch_khalid_al_mazrouei": CoachClaims(years_experience=_claim(
        4, "four years in, that background with young players remains central")),
}

OVERRIDES = {
    "courts": COURT_CLAIMS,
    "branches": BRANCH_CLAIMS,
    "classes": CLASS_CLAIMS,
    "packages": PACKAGE_CLAIMS,
    "coaches": COACH_CLAIMS,
}

FALSE_POSITIVES = {"crt_alquoz_sc03", "crt_yas_sc02", "crt_khalifa_sc01",
                    "crt_majaz_sc01", "crt_ajman_sc02"}


def _stub_extractor(entity: str, record: dict, text: str):
    schema = CLAIM_MODELS[entity]
    return OVERRIDES.get(entity, {}).get(record["id"], schema())


@pytest.fixture
def result():
    data = _load_data()
    ledger = Ledger()
    run_rule_checks(data, ledger)
    claims = collect_claims(data, _stub_extractor, ledger)
    adjudicate(data, claims, ledger)
    resolve(data, ledger)
    problems = verify_post_fix(data, ledger)
    return data, ledger, problems


def test_verify_post_fix_reports_no_problems(result):
    _, _, problems = result
    assert problems == []


def test_false_positive_courts_are_not_flipped(result):
    data, _, _ = result
    raw = {c["id"]: c["type"] for c in json.loads((ROOT / "catalog" / "courts.json").read_text())}
    courts = {c["id"]: c["type"] for c in data["courts"]}
    for cid in FALSE_POSITIVES:
        assert courts[cid] == raw[cid], f"{cid} should not have been flipped"


def test_noise_gated_courts_produce_no_claim(result):
    _, ledger, _ = result
    noisy = {i.entity_id for i in ledger.issues if i.issue_type == "unusable_text"}
    assert {"crt_alquoz_sc03", "crt_majaz_sc01"} <= noisy


def test_alain_sc04_is_corrected_to_outdoor(result):
    data, _, _ = result
    courts = {c["id"]: c["type"] for c in data["courts"]}
    assert courts["crt_alain_sc04"] == "outdoor"


def test_every_auto_fix_carries_evidence(result):
    _, ledger, _ = result
    for i in ledger.issues:
        if i.action == Action.AUTO_FIXED:
            assert i.evidence, i


def test_court_prices_are_all_resolved(result):
    data, _, _ = result
    for c in data["courts"]:
        assert c["price_per_hour_aed"] not in (None, 0, -1, 9999, 99999)

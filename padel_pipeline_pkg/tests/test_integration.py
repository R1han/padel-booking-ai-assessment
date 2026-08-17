"""In-process integration test against the real shipped data.

No API key and no subprocess: claims are hand-authored the way test_adjudicate.py
authors them, standing in for what the LLM would extract from these specific
descriptions (each evidence string below is a verbatim quote from the real
catalog text). The fixture is routed through the real collect_claims, so the
real prose gate (is_prose) decides what happens to the two noise-description
courts — they reach their unflipped state by the actual mechanism, not by
fixture fiat. This exercises the actual reconciliation pipeline —
collect_claims -> adjudicate -> resolve -> verify_post_fix — against the real
data, rather than a weaker offline stand-in for it. run_pipeline.py's own
wiring (arg parsing, writing catalog_clean/structured_clean, the
LLM-required hard-fail) is verified by running it for real in the next task,
against a human-reviewed diff.

The records covered here are exactly the ones that matter: the eight
court/description type disputes (two of which are noise-description courts —
see COURT_CLAIMS below), every branch's stated indoor/outdoor split, the one
shipped inverted-age class, the one sentinel-priced package, and the three
coaches whose shipped years_experience is implausible (87) but whose bios
state the real figure. Every other record's stub-extractor call returns an
empty claim object — "text says nothing" is the honest default collect_claims
would also produce for anything the LLM declines to extract.
"""
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
    """Read-only: catalog/ and structured/ are the source of truth and are never written to."""
    data = {}
    for name in CATALOG + STRUCTURED:
        folder = "catalog" if name in CATALOG else "structured"
        data[name] = json.loads((ROOT / folder / f"{name}.json").read_text())
    return data


def _claim(value, evidence: str, confidence: float = 0.9) -> dict:
    return {"value": value, "stated": True, "evidence": evidence, "confidence": confidence}


# -- courts: the eight description/type disputes in the shipped data -------
# crt_alquoz_sc03 and crt_majaz_sc01 are two of the three noise-description
# courts (crt_rak_rc02, the third, isn't disputed). They are deliberately
# absent from this dict: collect_claims runs the real is_prose gate on every
# record before it ever calls the stub extractor below, and their shipped
# descriptions fail that gate (confirmed against test_prose.py's frozen
# KNOWN_NOISE set) — so the stub is never even asked for a claim on them, and
# they get an "unusable_text" ledger issue plus an empty CourtClaims() by the
# same real code path a live LLM extractor would go through. That's the
# mechanism test_noise_gated_courts_produce_no_claim below checks.
COURT_CLAIMS = {
    "crt_jvc_pc04": CourtClaims(type=_claim(
        "outdoor", "An open-air court with nothing overhead but sky")),
    "crt_jvc_sc01": CourtClaims(type=_claim(
        "indoor", "A fully enclosed, air-conditioned court with a sealed roof")),
    "crt_yas_sc02": CourtClaims(type=_claim(
        "indoor", "A fully enclosed, air-conditioned court with a sealed roof")),
    # Real prose (passes the prose gate) but generic marketing boilerplate, not
    # a genuine description of the court — it only carries the bare word
    # "indoor" among a keyword list. A real extractor could plausibly still
    # pull this as a low-confidence claim; either way it must not win, because
    # br_khalifa's stated split already matches the shipped data exactly.
    "crt_khalifa_sc01": CourtClaims(type=_claim("indoor", "indoor", confidence=0.65)),
    "crt_alain_sc04": CourtClaims(type=_claim(
        "outdoor", "An open-air court with nothing overhead but sky")),
    "crt_ajman_sc02": CourtClaims(type=_claim(
        "indoor", "A fully enclosed, air-conditioned court with a sealed roof")),
}

# -- branches: every stated indoor/outdoor split, verbatim from the real prose --
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

# -- the one shipped inverted-age class (min_age=21, max_age=15) ------------
CLASS_CLAIMS = {
    "cls_junior_academy_yas": ClassClaims(
        min_age=_claim(9, "aged nine to fifteen"),
        max_age=_claim(15, "aged nine to fifteen")),
}

# -- the one sentinel-priced package (price_aed=99999) -----------------------
PACKAGE_CLAIMS = {
    "pkg_prime_evening_10_pack": PackageClaims(price_aed=_claim(1450, "AED 1450")),
}

# -- the three coaches shipped with an implausible years_experience (87) ----
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

# The branch-split constraint must veto every one of these five, either because
# the court's own description never produces a claim (the two noise courts) or
# because flipping it would break its branch's stated indoor/outdoor split.
FALSE_POSITIVES = {"crt_alquoz_sc03", "crt_yas_sc02", "crt_khalifa_sc01",
                    "crt_majaz_sc01", "crt_ajman_sc02"}


def _stub_extractor(entity: str, record: dict, text: str):
    """Stands in for llm.Extractor.extract: same signature, no network call.

    collect_claims only reaches this for records whose text already passed
    the real is_prose gate, so this never needs to reimplement that check —
    it just answers "what would the LLM have found" for the records this
    fixture hand-authors, and "nothing" for everything else.
    """
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
    """crt_alquoz_sc03 and crt_majaz_sc01 are absent from COURT_CLAIMS on purpose:
    this confirms it's the real is_prose gate, not the fixture, keeping them
    unflipped — collect_claims never reaches the stub extractor for either."""
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

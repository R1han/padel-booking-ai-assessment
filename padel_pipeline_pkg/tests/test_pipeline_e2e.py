"""End-to-end run on the real shipped data, heuristics only (no API key)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent

# Courts the offline (--no-llm) pipeline must not flip, and why each is protected:
#   crt_alquoz_sc03, crt_majaz_sc01 — descriptions are injected noise; the prose
#     gate excludes them, so no claim is ever produced.
#   crt_ajman_sc02 — the branch-split constraint vetoes it: br_ajman's description
#     states "two indoor and four outdoor", and the flip would make it 3/3.
# Deliberately absent: crt_yas_sc02 and crt_khalifa_sc01. The offline heuristic
# parses a branch split for only three of the eight branches (alquoz, jvc, ajman),
# so br_yas and br_khalifa state nothing the constraint can act on and their flips
# are correctly not vetoed. Asserting otherwise would assert a guarantee offline
# mode does not make. crt_jvc_sc01 is absent too: JVC's two proposed flips offset
# each other, leaving the branch tally unchanged, so the constraint has no opinion.
FALSE_FLIPS = {"crt_alquoz_sc03", "crt_majaz_sc01", "crt_ajman_sc02"}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    src = tmp_path_factory.mktemp("in")
    out = tmp_path_factory.mktemp("out")
    for d in ("catalog", "structured"):
        for f in (ROOT / d).glob("*.json"):
            shutil.copy(f, src / f.name)
    proc = subprocess.run(
        [sys.executable, "run_pipeline.py", "--input", str(src),
         "--output", str(out), "--no-llm"],
        cwd=PKG, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def test_writes_both_cleaned_directories(run):
    assert len(list((run / "catalog_clean").glob("*.json"))) == 7
    assert len(list((run / "structured_clean").glob("*.json"))) == 4


def test_no_court_is_flipped_against_its_branch_stated_split(run):
    courts = json.loads((run / "catalog_clean" / "courts.json").read_text())
    raw = {c["id"]: c["type"] for c in json.loads((ROOT / "catalog" / "courts.json").read_text())}
    changed = {c["id"] for c in courts if c["type"] != raw[c["id"]]}
    assert changed & FALSE_FLIPS == set()


def test_noisy_descriptions_never_change_a_field(run):
    ledger = json.loads((run / "issue_ledger.json").read_text())
    noisy = [i for i in ledger if i["issue_type"] == "unusable_text"]
    assert {i["entity_id"] for i in noisy} >= {"crt_alquoz_sc03", "crt_majaz_sc01", "crt_rak_rc02"}
    assert all(i["action"] == "quarantined" for i in noisy)


def test_every_auto_fix_carries_evidence(run):
    ledger = json.loads((run / "issue_ledger.json").read_text())
    for i in ledger:
        if i["action"] == "auto_fixed":
            assert i["evidence"], i


def test_court_prices_are_all_resolved(run):
    courts = json.loads((run / "catalog_clean" / "courts.json").read_text())
    for c in courts:
        assert c["price_per_hour_aed"] not in (None, 0, -1, 9999, 99999)

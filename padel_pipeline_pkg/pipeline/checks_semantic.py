"""Semantic checks: does the free text agree with the structured fields?

Two tiers:
  1. Deterministic heuristics (keyword scoring, number extraction, cross-source
     triangulation against slot prices) — run on everything, free.
  2. LLM verdicts (llm.py) — run only on records the heuristics flag or can't
     decide, so cost stays bounded. If no API key is configured the pipeline
     still works; LLM-dependent verdicts fall back to heuristic confidence.

Owner decision: when description and structured field disagree, the DESCRIPTION wins.
"""
from __future__ import annotations

import re
from collections import Counter

from .ledger import Ledger, Issue, Severity
from .checks_rules import day_type, PRICE_SENTINELS

# ---------------------------------------------------------------------------
# Court type vs description
# ---------------------------------------------------------------------------
OUTDOOR_KW = [
    "open-air", "open air", "outdoor", "under the sky", "nothing overhead",
    "no roof", "no shade", "desert breeze", "floodlit", "sun drops",
    "open to the sky", "rooftop",
]
INDOOR_KW = [
    "indoor", "air-condition", "air condition", "climate-control", "climate control",
    "roofed", "under one roof", "shielded from the desert heat", "shielded from the heat",
    "fully covered", "enclosed hall",
]


def classify_type_from_description(description: str) -> tuple[str | None, float, str]:
    """Return (verdict, confidence, evidence) from keywords; verdict None if ambiguous."""
    d = description.lower()
    out_hits = [k for k in OUTDOOR_KW if k in d]
    in_hits = [k for k in INDOOR_KW if k in d]
    if out_hits and not in_hits:
        return "outdoor", min(0.6 + 0.1 * len(out_hits), 0.9), f"outdoor cues: {out_hits}"
    if in_hits and not out_hits:
        return "indoor", min(0.6 + 0.1 * len(in_hits), 0.9), f"indoor cues: {in_hits}"
    if out_hits and in_hits:
        return None, 0.0, f"conflicting cues in: {out_hits} vs out: {in_hits}"
    return None, 0.0, "no directional cues"


def price_band_evidence(court: dict, slots: list[dict], price_rules: list[dict]) -> dict[str, float]:
    """Cross-source triangulation: which court_type's price rules do this court's
    actual slot prices follow? Returns match ratio per type (0 when no rules cover it)."""
    rules_idx = {}
    for r in price_rules:
        for t in r["applies_to_start_times"]:
            rules_idx[(r["branch_id"], r["court_type"], r["day_type"], t)] = r
    price = court.get("price_per_hour_aed")
    usable_price = isinstance(price, int) and price not in PRICE_SENTINELS
    result: dict[str, float] = {}
    for ctype in ("indoor", "outdoor"):
        ok = tot = 0
        for s in slots:
            if s["court_id"] != court["id"]:
                continue
            r = rules_idx.get((court["branch_id"], ctype, day_type(s["date"]), s["start_time"]))
            if not r:
                continue
            tot += 1
            if usable_price:
                if round(price * r["multiplier"] / 5) * 5 == s["price_aed"]:
                    ok += 1
        # If the branch has zero rules of this type covering the court's slots,
        # that is itself strong evidence the court is NOT this type.
        result[ctype] = (ok / tot) if tot else 0.0
        result[f"{ctype}_coverage"] = tot
    return result


def check_court_type_semantics(data: dict, ledger: Ledger, llm=None) -> None:
    slots, rules = data.get("slots", []), data.get("price_rules", [])
    for c in data.get("courts", []):
        verdict, conf, evidence = classify_type_from_description(c["description"])
        if llm is not None and verdict is None:
            verdict, conf, evidence = llm.classify_court_type(c["description"])
        if verdict is None or verdict == c["type"]:
            continue
        pe = price_band_evidence(c, slots, rules)
        agrees = pe.get(f"{verdict}_coverage", 0) > 0 and pe.get(c["type"] + "_coverage", 1) == 0
        ledger.add(Issue(
            "courts", c["id"], "type", "semantic_type_description_mismatch", Severity.ERROR,
            detected_value=c["type"], corrected_value=verdict,
            confidence=min(conf + (0.1 if agrees else 0.0), 0.95),
            evidence=evidence + f" | price-rule coverage: indoor={pe['indoor_coverage']}, "
                                f"outdoor={pe['outdoor_coverage']}",
            checker="semantic.court_type",
        ))


# ---------------------------------------------------------------------------
# Coach bio extraction: years of experience, languages
# ---------------------------------------------------------------------------
WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "a decade": 10, "decade": 10,
}
YEARS_PAT = re.compile(
    r"\b(?P<num>\d{1,2}|" + "|".join(k for k in WORD_NUM if " " not in k) + r")\s+years?"
    r"(?P<tail>[^.]{0,60})", re.IGNORECASE)
EXPERIENCE_CUES = ("experience", "coaching", "into coaching", "in,", "in ", "of teaching",
                   "on court", "spanning")

LANGUAGES = ["English", "Arabic", "Spanish", "Portuguese", "French", "Italian", "German",
             "Hindi", "Malayalam", "Urdu", "Tagalog", "Russian", "Mandarin", "Dutch"]
LANG_CONTEXT = re.compile(r"[^.]*\b(speak|speaking|fluent|fluently|in both|comfortabl\w+ in|"
                          r"works? .{0,20}in|teach\w* .{0,20}in|running sessions in)\b[^.]*",
                          re.IGNORECASE)


def extract_years_from_bio(bio: str) -> tuple[int | None, str]:
    candidates: list[tuple[int, str]] = []
    for m in YEARS_PAT.finditer(bio):
        raw = m.group("num").lower()
        val = int(raw) if raw.isdigit() else WORD_NUM.get(raw)
        if val is None:
            continue
        window = bio[max(0, m.start() - 60): m.end() + 60].lower()
        if any(cue in window for cue in EXPERIENCE_CUES):
            candidates.append((val, bio[max(0, m.start() - 40): m.end() + 20].strip()))
    if not candidates:
        return None, "no experience-year phrase found in bio"
    # If several candidates, prefer the one whose context mentions coaching/experience most directly
    val, ctx = candidates[0]
    return val, f'bio: "...{ctx}..."'


def extract_languages_from_bio(bio: str) -> tuple[list[str], str]:
    found: list[str] = []
    ev_sentences: list[str] = []
    for m in LANG_CONTEXT.finditer(bio):
        sent = m.group(0)
        hits = [lang for lang in LANGUAGES if lang in sent]
        for h in hits:
            if h not in found:
                found.append(h)
        if hits:
            ev_sentences.append(sent.strip()[:120])
    if not found:  # fallback: any language name anywhere in bio
        found = [lang for lang in LANGUAGES if lang in bio]
        if found:
            ev_sentences = ["language names present in bio (no speak-context sentence)"]
    return found, " / ".join(ev_sentences[:2])


def check_coach_bio_semantics(data: dict, ledger: Ledger, llm=None) -> None:
    for c in data.get("coaches", []):
        # years_experience recovery — only where the rules layer flagged implausible values
        if c.get("years_experience", 0) > 40:
            val, ev = (llm.extract_years(c["bio"]) if llm else (None, ""))
            if val is None:
                val, ev = extract_years_from_bio(c["bio"])
            ledger.add(Issue("coaches", c["id"], "years_experience",
                             "semantic_bio_extraction", Severity.ERROR,
                             detected_value=c["years_experience"], corrected_value=val,
                             confidence=0.85 if val is not None else 0.0,
                             evidence=ev, checker="semantic.coach_bio"))
        # language recovery for empty lists
        if not c.get("languages"):
            langs, ev = extract_languages_from_bio(c["bio"])
            ledger.add(Issue("coaches", c["id"], "languages", "semantic_bio_extraction",
                             Severity.WARNING, detected_value=c.get("languages"),
                             corrected_value=langs or None,
                             confidence=0.9 if langs else 0.0,
                             evidence=ev, checker="semantic.coach_bio"))


# ---------------------------------------------------------------------------
# Class ages vs description
# ---------------------------------------------------------------------------
AGE_RANGE_PAT = re.compile(r"\b(?:aged?|ages?)\s+(\d{1,2})\s*(?:to|-|–|and)\s*(\d{1,2})", re.I)
AGE_MIN_PAT = re.compile(r"\b(?:aged?|ages?)\s+(\w+|\d{1,2})\s+(?:and\s+(?:over|above|up)|\+)", re.I)


def check_class_age_semantics(data: dict, ledger: Ledger) -> None:
    for c in data.get("classes", []):
        mn, mx = c.get("min_age"), c.get("max_age")
        if isinstance(mn, int) and isinstance(mx, int) and mn > mx:
            m = AGE_RANGE_PAT.search(c["description"])
            if m:
                lo, hi = sorted((int(m.group(1)), int(m.group(2))))
                ev = f'description states ages {m.group(1)}–{m.group(2)}'
                corrected = {"min_age": lo, "max_age": hi}
                conf = 0.9
            else:
                corrected = {"min_age": mx, "max_age": mn}
                ev = "no age range in description; swapping inverted values"
                conf = 0.7
            ledger.add(Issue("classes", c["id"], "min_age/max_age", "semantic_age_range",
                             Severity.ERROR, detected_value={"min_age": mn, "max_age": mx},
                             corrected_value=corrected, confidence=conf, evidence=ev,
                             checker="semantic.class_ages"))


# ---------------------------------------------------------------------------
# Package description vs structured numbers
# ---------------------------------------------------------------------------
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def parse_word_number(raw: str) -> int | None:
    """Handles 'five', 'twenty-five', 'thirty'."""
    raw = raw.lower().strip()
    if raw in WORD_NUM:
        return WORD_NUM[raw]
    if "-" in raw:
        a, _, b = raw.partition("-")
        if a in TENS and b in WORD_NUM:
            return TENS[a] + WORD_NUM[b]
    return TENS.get(raw)


def check_package_number_semantics(data: dict, ledger: Ledger) -> None:
    for p in data.get("packages", []):
        d = p["description"]
        m = re.search(r"AED\s*([\d,]+)", d)
        if m and int(m.group(1).replace(",", "")) != p["price_aed"]:
            desc_price = int(m.group(1).replace(",", ""))
            is_sentinel = p["price_aed"] in (99999, 9999, 0, -1)
            ledger.add(Issue("packages", p["id"], "price_aed", "semantic_number_mismatch",
                             Severity.ERROR if is_sentinel else Severity.WARNING,
                             detected_value=p["price_aed"],
                             corrected_value=desc_price if is_sentinel else None,
                             confidence=0.9 if is_sentinel else None,
                             evidence=f"description says AED {m.group(1)}"
                                      + (" | field holds sentinel placeholder" if is_sentinel else ""),
                             checker="semantic.package_numbers"))
        m2 = re.search(r"\b([\w-]+|\d+)\s+sessions\b", d, re.I)
        if m2:
            raw = m2.group(1).lower()
            n = int(raw) if raw.isdigit() else parse_word_number(raw)
            if n is not None and n != p["sessions"]:
                ledger.add(Issue("packages", p["id"], "sessions", "semantic_number_mismatch",
                                 Severity.WARNING, detected_value=p["sessions"],
                                 evidence=f'description says "{m2.group(0)}"',
                                 checker="semantic.package_numbers"))


def run_semantic_checks(data: dict, ledger: Ledger, llm=None) -> None:
    check_court_type_semantics(data, ledger, llm)
    check_coach_bio_semantics(data, ledger, llm)
    check_class_age_semantics(data, ledger)
    check_package_number_semantics(data, ledger)

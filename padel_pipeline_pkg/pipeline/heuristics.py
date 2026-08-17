"""Deterministic claim extraction for runs without an API key.

Covers the subset the original pipeline handled — court type, coach years and
languages, class age ranges, package numbers. Every other field is returned
unstated, which is honest: the heuristics genuinely do not know.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .checks_semantic import (
    AGE_RANGE_PAT,
    classify_type_from_description,
    extract_languages_from_bio,
    extract_years_from_bio,
    parse_word_number,
)
from .claims import CLAIM_MODELS


def _set(obj: BaseModel, attr: str, value, evidence: str, confidence: float) -> None:
    claim = getattr(obj, attr)
    claim.value = value
    claim.stated = True
    claim.evidence = evidence
    claim.confidence = confidence


def extract(entity: str, record: dict, text: str) -> BaseModel:
    out = CLAIM_MODELS[entity]()

    if entity == "courts":
        verdict, conf, evidence = classify_type_from_description(text)
        if verdict:
            _set(out, "type", verdict, evidence, conf)

    elif entity == "coaches":
        years, evidence = extract_years_from_bio(text)
        if years is not None:
            _set(out, "years_experience", years, evidence, 0.85)
        langs, lang_evidence = extract_languages_from_bio(text)
        if langs:
            _set(out, "languages", langs, lang_evidence, 0.9)

    elif entity == "classes":
        m = AGE_RANGE_PAT.search(text)
        if m:
            lo, hi = sorted((int(m.group(1)), int(m.group(2))))
            _set(out, "min_age", lo, m.group(0), 0.9)
            _set(out, "max_age", hi, m.group(0), 0.9)

    elif entity == "packages":
        m = re.search(r"AED\s*([\d,]+)", text)
        if m:
            _set(out, "price_aed", int(m.group(1).replace(",", "")), m.group(0), 0.9)
        m2 = re.search(r"\b([\w-]+|\d+)\s+sessions\b", text, re.I)
        if m2:
            raw = m2.group(1).lower()
            n = int(raw) if raw.isdigit() else parse_word_number(raw)
            if n is not None:
                _set(out, "sessions", n, m2.group(0), 0.85)

    elif entity == "branches":
        m = re.search(r"\b(\w+)\s+(?:courts?\s+)?indoors?\s+and\s+(\w+)\s+outdoors?\b", text, re.I)
        if m:
            a, b = parse_word_number(m.group(1)), parse_word_number(m.group(2))
            if a is not None and b is not None:
                _set(out, "indoor_courts", a, m.group(0), 0.8)
                _set(out, "outdoor_courts", b, m.group(0), 0.8)

    return out

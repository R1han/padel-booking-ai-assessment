import os

from pipeline.claims import CourtClaims
from pipeline.llm import build_prompt, get_llm, verify_evidence

TEXT = ("An open-air court with nothing overhead but sky - floodlit for evening "
        "play and cooled by the desert breeze once the sun drops.")


def test_get_llm_is_none_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_llm() is None


def test_prompt_names_the_entity_and_carries_the_text():
    p = build_prompt("courts", TEXT)
    assert TEXT in p
    assert "court" in p.lower()
    assert "not explicitly stated" in p.lower()


def test_verify_evidence_keeps_a_quotable_claim():
    c = CourtClaims(type={"value": "outdoor", "stated": True,
                          "evidence": "nothing overhead but sky", "confidence": 0.95})
    out = verify_evidence(c, TEXT)
    assert out.type.stated is True


def test_verify_evidence_downgrades_an_unquotable_claim():
    c = CourtClaims(type={"value": "indoor", "stated": True,
                          "evidence": "air-conditioned hall", "confidence": 0.9})
    out = verify_evidence(c, TEXT)
    assert out.type.stated is False
    assert "not found in source text" in out.type.evidence


def test_verify_evidence_downgrades_an_empty_quote():
    c = CourtClaims(type={"value": "indoor", "stated": True,
                          "evidence": "", "confidence": 0.9})
    assert verify_evidence(c, TEXT).type.stated is False

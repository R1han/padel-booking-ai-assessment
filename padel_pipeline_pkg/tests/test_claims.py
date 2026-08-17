import pytest
from pydantic import ValidationError

from pipeline.claims import (
    AUTHORITY, CLAIM_MODELS, FIELD_MAP, TEXT_FIELD,
    CourtClaims, FieldClaim,
)


def test_default_claim_is_unstated():
    c = FieldClaim()
    assert c.stated is False
    assert c.value is None
    assert c.evidence == ""


def test_enum_field_rejects_vocabulary_outside_the_dataset():
    with pytest.raises(ValidationError):
        CourtClaims(type={"value": "semi_outdoor", "stated": True,
                          "evidence": "x", "confidence": 0.9})


def test_enum_field_accepts_canonical_value():
    c = CourtClaims(type={"value": "outdoor", "stated": True,
                          "evidence": "nothing overhead but sky", "confidence": 0.95})
    assert c.type.value == "outdoor"
    assert c.surface.stated is False  # untouched fields default to unstated


def test_every_entity_has_a_text_field_and_a_schema():
    assert set(CLAIM_MODELS) == set(TEXT_FIELD)
    assert TEXT_FIELD["coaches"] == "bio"
    assert TEXT_FIELD["courts"] == "description"


def test_every_claim_attr_maps_to_a_record_field_with_an_authority():
    for entity, model in CLAIM_MODELS.items():
        for attr in model.model_fields:
            record_field = FIELD_MAP[(entity, attr)]
            assert (entity, record_field) in AUTHORITY, f"{entity}.{record_field}"


def test_authority_values_are_the_three_known_policies():
    assert set(AUTHORITY.values()) <= {"description", "structured", "quarantine"}


def test_transactional_fields_defer_to_the_structured_record():
    assert AUTHORITY[("packages", "price_aed")] == "structured"
    assert AUTHORITY[("coaches", "rate_per_hour_aed")] == "structured"


def test_descriptive_fields_defer_to_the_description():
    assert AUTHORITY[("courts", "type")] == "description"
    assert AUTHORITY[("courts", "walls")] == "description"
    assert AUTHORITY[("coaches", "languages")] == "description"


def test_identity_and_foreign_keys_quarantine_on_conflict():
    assert AUTHORITY[("classes", "coach_id")] == "quarantine"
    assert AUTHORITY[("coaches", "branch_id")] == "quarantine"

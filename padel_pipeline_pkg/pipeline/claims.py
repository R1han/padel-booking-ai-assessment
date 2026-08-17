"""Claim schemas and the per-field authority policy.

Pure data. The LLM fills these in; adjudicate.py decides what to do with them.

Two rules the extraction prompt enforces and the schema encodes:
  * a field not explicitly stated in the text stays `stated=False`
  * a stated field carries a verbatim quote in `evidence`

`value` and `stated` are both needed because "the text does not mention an
upper age limit" and "the text says there is no upper age limit" are different
facts, and this dataset contains both.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FieldClaim(BaseModel):
    value: str | int | float | list[str] | None = None
    stated: bool = False
    evidence: str = Field(default="", description="Verbatim quote from the source text")
    confidence: float = 0.0


# -- enum-constrained claims -------------------------------------------------
class TypeClaim(FieldClaim):
    value: Literal["indoor", "outdoor"] | None = None


class SurfaceClaim(FieldClaim):
    value: Literal["sand_filled_turf", "textured_turf", "artificial_grass"] | None = None


class WallsClaim(FieldClaim):
    value: Literal["panoramic_glass", "concrete_and_glass"] | None = None


class LightingClaim(FieldClaim):
    value: Literal["LED", "floodlight", "LED_dimmable"] | None = None


class LevelClaim(FieldClaim):
    value: Literal["beginner", "intermediate", "advanced", "all_levels"] | None = None


class GenderClaim(FieldClaim):
    value: Literal["mixed", "ladies_only"] | None = None


class LevelFocusClaim(FieldClaim):
    value: Literal["beginner", "beginner_to_intermediate", "intermediate_to_advanced",
                   "advanced", "all_levels"] | None = None


class IntClaim(FieldClaim):
    value: int | None = None


class StrClaim(FieldClaim):
    value: str | None = None


class ListClaim(FieldClaim):
    value: list[str] | None = None


# -- per-entity schemas ------------------------------------------------------
class CourtClaims(BaseModel):
    name: StrClaim = StrClaim()
    type: TypeClaim = TypeClaim()
    surface: SurfaceClaim = SurfaceClaim()
    walls: WallsClaim = WallsClaim()
    lighting: LightingClaim = LightingClaim()


class CoachClaims(BaseModel):
    name: StrClaim = StrClaim()
    branch_name: StrClaim = StrClaim()
    years_experience: IntClaim = IntClaim()
    languages: ListClaim = ListClaim()
    level_focus: LevelFocusClaim = LevelFocusClaim()
    specialties: ListClaim = ListClaim()
    rate_per_hour_aed: IntClaim = IntClaim()


class ClassClaims(BaseModel):
    name: StrClaim = StrClaim()
    branch_name: StrClaim = StrClaim()
    coach_name: StrClaim = StrClaim()
    level: LevelClaim = LevelClaim()
    gender: GenderClaim = GenderClaim()
    min_age: IntClaim = IntClaim()
    max_age: IntClaim = IntClaim()
    price_per_term_aed: IntClaim = IntClaim()


class PackageClaims(BaseModel):
    name: StrClaim = StrClaim()
    price_aed: IntClaim = IntClaim()
    sessions: IntClaim = IntClaim()
    branch_names: ListClaim = ListClaim()


class BranchClaims(BaseModel):
    name: StrClaim = StrClaim()
    emirate: StrClaim = StrClaim()
    area: StrClaim = StrClaim()
    court_count: IntClaim = IntClaim()
    indoor_courts: IntClaim = IntClaim()
    outdoor_courts: IntClaim = IntClaim()
    amenities: ListClaim = ListClaim()


CLAIM_MODELS: dict[str, type[BaseModel]] = {
    "courts": CourtClaims,
    "coaches": CoachClaims,
    "classes": ClassClaims,
    "packages": PackageClaims,
    "branches": BranchClaims,
}

TEXT_FIELD: dict[str, str] = {
    "courts": "description",
    "coaches": "bio",
    "classes": "description",
    "packages": "description",
    "branches": "description",
}

# (entity, claim attribute) -> the record field it is a claim about.
# Name-valued claims map to the id field they resolve to; adjudicate.py does
# the name -> id resolution before comparing.
FIELD_MAP: dict[tuple[str, str], str] = {
    ("courts", "name"): "name",
    ("courts", "type"): "type",
    ("courts", "surface"): "surface",
    ("courts", "walls"): "walls",
    ("courts", "lighting"): "lighting",
    ("coaches", "name"): "name",
    ("coaches", "branch_name"): "branch_id",
    ("coaches", "years_experience"): "years_experience",
    ("coaches", "languages"): "languages",
    ("coaches", "level_focus"): "level_focus",
    ("coaches", "specialties"): "specialties",
    ("coaches", "rate_per_hour_aed"): "rate_per_hour_aed",
    ("classes", "name"): "name",
    ("classes", "branch_name"): "branch_id",
    ("classes", "coach_name"): "coach_id",
    ("classes", "level"): "level",
    ("classes", "gender"): "gender",
    ("classes", "min_age"): "min_age",
    ("classes", "max_age"): "max_age",
    ("classes", "price_per_term_aed"): "price_per_term_aed",
    ("packages", "name"): "name",
    ("packages", "price_aed"): "price_aed",
    ("packages", "sessions"): "sessions",
    ("packages", "branch_names"): "branch_ids",
    ("branches", "name"): "name",
    ("branches", "emirate"): "emirate",
    ("branches", "area"): "area",
    ("branches", "court_count"): "court_count",
    ("branches", "indoor_courts"): "court_count",   # corroborates the split, not a field
    ("branches", "outdoor_courts"): "court_count",
    ("branches", "amenities"): "amenities",
}

# What wins when the claim and the structured value disagree.
#   description — descriptive facts the prose is authored to convey
#   structured  — transactional facts (money, dates) the record exists to hold
#   quarantine  — identity and foreign keys; a mismatch means one record is
#                 wrong and neither side can say which
AUTHORITY: dict[tuple[str, str], str] = {
    ("courts", "type"): "description",
    ("courts", "surface"): "description",
    ("courts", "walls"): "description",
    ("courts", "lighting"): "description",
    ("courts", "name"): "quarantine",
    ("coaches", "years_experience"): "description",
    ("coaches", "languages"): "description",
    ("coaches", "level_focus"): "description",
    ("coaches", "specialties"): "description",
    ("coaches", "rate_per_hour_aed"): "structured",
    ("coaches", "branch_id"): "quarantine",
    ("coaches", "name"): "quarantine",
    ("classes", "level"): "description",
    ("classes", "gender"): "description",
    ("classes", "min_age"): "description",
    ("classes", "max_age"): "description",
    ("classes", "price_per_term_aed"): "structured",
    ("classes", "coach_id"): "quarantine",
    ("classes", "branch_id"): "quarantine",
    ("classes", "name"): "quarantine",
    ("packages", "price_aed"): "structured",
    ("packages", "sessions"): "structured",
    ("packages", "branch_ids"): "quarantine",
    ("packages", "name"): "quarantine",
    ("branches", "court_count"): "description",
    ("branches", "emirate"): "description",
    ("branches", "area"): "description",
    ("branches", "amenities"): "description",
    ("branches", "name"): "quarantine",
}

"""Pydantic schemas: the structural contract each record must meet before deeper checks run.

Validation is deliberately tolerant on fields we know are dirty (e.g. nullable prices,
sentinel values): those pass the schema and are handled by the rule/semantic layers,
which can record *why* and fix with evidence. The schema layer only rejects records
that are structurally unusable (wrong types, missing ids, malformed dates).
"""
from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field, field_validator

DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
TIME_RE = r"^\d{2}:\d{2}$"


class Coordinates(BaseModel):
    lat: float
    lng: float


class Branch(BaseModel):
    id: str
    name: str
    emirate: str
    area: str
    coordinates: Coordinates | None = None
    description: str
    amenities: list[str]
    opening_hours: dict | str
    phone: str
    court_count: int = Field(ge=0)


class Court(BaseModel):
    id: str
    code: str
    branch_id: str
    name: str
    type: str                       # enum checked in rules layer (so bad values are ledgered, not dropped)
    surface: str
    walls: str
    lighting: str
    description: str
    price_per_hour_aed: int | None = None   # nulls/sentinels handled by fixer


class Coach(BaseModel):
    id: str
    name: str
    branch_id: str
    bio: str
    specialties: list[str]
    languages: list[str]
    level_focus: list[str] | str
    years_experience: int = Field(ge=0)
    rate_per_hour_aed: int = Field(gt=0)
    internal_phone: str | None = None
    internal_email: str | None = None


class PadelClass(BaseModel):
    id: str
    name: str
    branch_id: str
    description: str
    level: str
    min_age: int | None = None
    max_age: int | None = None
    gender: str
    schedule: dict | list | str
    price_per_term_aed: int = Field(gt=0)
    coach_id: str


class Package(BaseModel):
    id: str
    name: str
    branch_ids: list[str]
    description: str
    price_aed: int = Field(gt=0)
    sessions: int = Field(gt=0)
    valid_from: str = Field(pattern=DATE_RE)
    valid_until: str = Field(pattern=DATE_RE)
    conditions: list[str] | str
    status: str


class Policy(BaseModel):
    id: str
    title: str
    body: str
    category: str


class PriceRule(BaseModel):
    id: str
    branch_id: str
    court_type: str
    day_type: str
    band: str
    applies_to_start_times: list[str]
    multiplier: float = Field(gt=0)
    base_price_aed: int = Field(gt=0)
    price_aed: int = Field(gt=0)

    @field_validator("applies_to_start_times")
    @classmethod
    def _times(cls, v: list[str]) -> list[str]:
        import re
        for t in v:
            if not re.match(TIME_RE, t):
                raise ValueError(f"bad time {t}")
        return v


class Review(BaseModel):
    id: str
    branch_id: str
    court_id: str | None = None
    coach_id: str | None = None
    rating: int = Field(ge=1, le=5)
    text: str
    author_name: str
    date: str = Field(pattern=DATE_RE)


class Booking(BaseModel):
    id: str
    slot_ids: list[str] = Field(min_length=1)
    user_id: str
    status: str
    created_at: str
    duration_min: int = Field(gt=0)


class CoachSchedule(BaseModel):
    id: str
    coach_id: str
    branch_id: str
    date: str = Field(pattern=DATE_RE)
    start_time: str = Field(pattern=TIME_RE)
    end_time: str = Field(pattern=TIME_RE)
    status: str


class Slot(BaseModel):
    id: str
    court_id: str
    branch_id: str
    date: str = Field(pattern=DATE_RE)
    start_time: str = Field(pattern=TIME_RE)
    duration_min: int = Field(gt=0)
    status: str
    price_aed: int = Field(gt=0)
    version: int


SCHEMAS: dict[str, type[BaseModel]] = {
    "branches": Branch,
    "courts": Court,
    "coaches": Coach,
    "classes": PadelClass,
    "packages": Package,
    "policies": Policy,
    "price_rules": PriceRule,
    "reviews": Review,
    "bookings": Booking,
    "coach_schedules": CoachSchedule,
    "slots": Slot,
}

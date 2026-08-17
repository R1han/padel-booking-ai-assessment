import json
from pathlib import Path

import pytest

from pipeline.prose import is_prose

ROOT = Path(__file__).resolve().parents[2]

# The six records whose "description" is injected scraper noise, not prose.
KNOWN_NOISE = {
    "crt_alquoz_sc03", "crt_majaz_sc01", "crt_rak_rc02",
    "cls_adult_beginner_course_jvc", "cls_junior_academy_jvc", "cls_cardio_padel_rak",
}


@pytest.mark.parametrize("text", [
    "JavaScript is required to view padel court booking availability. "
    "Please enable JavaScript and reload. Error code BK-500. "
    "padel booking court indoor coach price ladies beginner",
    "Home > Branches > Padel > Court Booking > Availability. Skip to main content. "
    "Menu: Book a court | Coaching | Prices | Indoor courts | Contact.",
    '<div class="court-listing"><span>Padel Court Booking</span>'
    "<ul><li>indoor court</li></ul></div>",
    "Your session has expired. Please log in again to complete your padel court booking.",
    "",
    None,
    "Short.",
])
def test_rejects_non_prose(text):
    usable, reason = is_prose(text)
    assert usable is False
    assert reason != "ok"


def test_accepts_real_description():
    text = (
        "Panoramic Court 1 sits behind full-height glass, sand-filled turf underfoot "
        "for a true, controlled bounce. LED lighting keeps play crisp at any hour, "
        "indoors and shielded from the desert heat. A favourite with players who value "
        "consistency in both surface and sightlines, from club nights to focused solo drills."
    )
    assert is_prose(text) == (True, "ok")


def test_gate_matches_known_noise_in_shipped_data():
    """Every noisy record is rejected and every other record is accepted."""
    rejected = set()
    for name, key in [("courts", "description"), ("classes", "description"),
                      ("coaches", "bio"), ("packages", "description"),
                      ("branches", "description")]:
        for r in json.loads((ROOT / "catalog" / f"{name}.json").read_text()):
            if not is_prose(r.get(key))[0]:
                rejected.add(r["id"])
    assert rejected == KNOWN_NOISE

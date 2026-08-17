from pipeline.heuristics import extract

COURT_TEXT = ("An open-air court with nothing overhead but sky - floodlit for "
              "evening play and cooled by the desert breeze once the sun drops.")
BIO = ("Diego Fernandez cut his teeth on the competitive circuit before a knee "
       "injury nudged him toward coaching fourteen years ago. Diego speaks "
       "English and Spanish fluently, switching between them mid-sentence.")


def test_court_type_extracted_from_prose():
    c = extract("courts", {"id": "x"}, COURT_TEXT)
    assert c.type.stated is True
    assert c.type.value == "outdoor"
    assert c.type.evidence


def test_coach_years_and_languages_extracted_from_bio():
    c = extract("coaches", {"id": "x"}, BIO)
    assert c.years_experience.value == 14
    assert c.languages.value == ["English", "Spanish"]


def test_fields_the_heuristics_do_not_cover_stay_unstated():
    c = extract("courts", {"id": "x"}, COURT_TEXT)
    assert c.surface.stated is False
    assert c.walls.stated is False


def test_never_returns_none():
    for entity in ("courts", "coaches", "classes", "packages", "branches"):
        assert extract(entity, {"id": "x"}, "Some text that says nothing useful. " * 6) is not None

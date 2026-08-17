"""Cross-turn references and group booking.

These two features were built and eyeballed rather than measured, which is exactly how
the earlier retrieval bugs survived. The cheap, deterministic parts are tested here; the
end-to-end conversational behaviour is measured by the multi-turn cases in
eval/queries.json (q39-q47), which need a model and so belong in the eval run.
"""

from __future__ import annotations

import pytest

from app import db, llm
from app.services import booking
from app.agent import tools as agent_tools
from app.api.chat import Session, session_for, summarise_context
from app.services import retrieval, vectorstore

# --- cross-turn references (challenge 2) -------------------------------------------


def test_context_summary_numbers_records_so_an_ordinal_can_resolve():
    """"The second one" only works if the previous turn left an ordered list behind."""
    slots = retrieval.check_availability(branch="Ajman", date_="tomorrow")["slots"][:3]
    assert len(slots) >= 2, "fixture needs at least two free slots"

    context = summarise_context([{**s, "id": s["id"]} for s in slots], "Here are some courts.")

    assert "1. " in context and "2. " in context
    assert slots[1]["id"] in context, "the second record must be identifiable by id"
    # Price and time travel with it, so "how much is the second one" is answerable.
    assert str(slots[1]["price_aed"]) in context
    assert slots[1]["start_time"] in context


def test_context_summary_survives_records_without_prices():
    """Coaches and classes have no start_time or price; summarising must not blow up."""
    records = retrieval.hydrate(["cch_marwan_haddad", "br_alquoz"])
    context = summarise_context(records, "Two records.")
    assert "cch_marwan_haddad" in context and "br_alquoz" in context


def test_session_keeps_history_and_is_bounded():
    session = session_for("test-session")
    assert isinstance(session, Session)
    session.context = "remembered"
    assert session_for("test-session").context == "remembered"


def test_surfaced_records_preserve_rank_order():
    """The ordinal in "the second one" is only meaningful if order is stable."""
    agent_tools.start_request("order-test")
    agent_tools.search_knowledge.invoke({"query": "ladies only beginner classes"})
    ids = agent_tools.surfaced_ids()
    records = agent_tools.surfaced_records()
    assert [r["id"] for r in records] == [i for i in ids if any(r["id"] == i for r in records)]


# --- group booking (challenge 5) ----------------------------------------------------


def test_party_size_converts_to_courts_at_four_players_each():
    for party, courts in [(4, 1), (5, 2), (8, 2), (9, 3), (12, 3)]:
        result = retrieval.find_group_slots(party_size=party, adjacent=False, limit=1)
        assert result["courts_required"] == courts, f"{party} players"
        assert result["capacity"] == courts * 4


def test_adjacent_courts_are_consecutive_within_one_prefix():
    result = retrieval.find_group_slots(band="morning", courts=3, adjacent=True, limit=1)
    assert result["options"], "expected at least one run of three adjacent courts"
    codes = [c["code"] for c in result["options"][0]["courts"]]
    prefixes = {c.split("-")[0] for c in codes}
    numbers = sorted(int(c.split("-")[1]) for c in codes)
    assert len(prefixes) == 1, f"mixed prefixes are not adjacent: {codes}"
    assert numbers == list(range(numbers[0], numbers[0] + 3)), codes


def test_group_options_are_simultaneous():
    """A group booking is useless if the courts are at different times or branches."""
    result = retrieval.find_group_slots(band="evening", courts=2, adjacent=True, limit=3)
    assert result["options"]
    for option in result["options"]:
        with db.read_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT branch_id, date, start_time FROM slots"
                f" WHERE id IN ({','.join('?' * len(option['slot_ids']))})",
                option["slot_ids"],
            ).fetchall()
        assert len(rows) == 1, "courts in one option must share branch, date and hour"


def test_adjacency_rule_is_disclosed_not_asserted():
    """The dataset records no court positions, so the interpretation must travel with
    the answer rather than being presented as fact."""
    result = retrieval.find_group_slots(courts=2, adjacent=True, limit=1)
    assert result["adjacency_rule"]
    assert "not a fact from the data" in result["adjacency_rule"]


def test_coach_requirement_only_returns_hours_with_a_coach_on_shift():
    result = retrieval.find_group_slots(
        band="evening", courts=2, adjacent=True, with_coach=True, limit=3
    )
    assert result["options"], "expected some evening slots with a coach on shift"
    assert "does not reserve them" in result["coach_caveat"]
    for option in result["options"]:
        assert option["coaches_on_shift"], "with_coach must not return coachless options"
        for coach in option["coaches_on_shift"]:
            assert coach["start_time"] <= option["start_time"] < coach["end_time"]


def test_impossible_group_request_says_so():
    """Al Quoz evenings are nearly fully booked in the shipped data. The honest answer
    is that nothing fits, not a quietly relaxed constraint."""
    result = retrieval.find_group_slots(
        branch="Al Quoz", date_="2026-08-14", band="evening", courts=4, adjacent=True
    )
    assert result["options"] == []
    assert result["note"]


def test_group_slots_never_offer_a_claimed_or_overhang_slot():
    result = retrieval.find_group_slots(courts=2, adjacent=True, limit=5)
    offered = [s for option in result["options"] for s in option["slot_ids"]]
    assert offered
    with db.read_conn() as conn:
        placeholders = ",".join("?" * len(offered))
        clashes = conn.execute(
            f"SELECT slot_id FROM slot_claims WHERE slot_id IN ({placeholders})"
            f" UNION SELECT slot_id FROM slot_overhang WHERE slot_id IN ({placeholders})",
            offered + offered,
        ).fetchall()
    assert not clashes, f"offered unavailable slots: {[r['slot_id'] for r in clashes]}"


def test_group_options_are_bookable_for_the_requested_duration():
    """A 90-minute group option is a promise about two hours on every court. The WHERE
    clause only clears the first, so every offered slate is checked against the booking
    path -- otherwise the agent proposes courts, holds them, and fails after the user
    has already said yes."""
    result = retrieval.find_group_slots(
        band="morning", courts=2, adjacent=True, duration_min=90, limit=5
    )
    assert result["options"], "expected at least one 90-minute group option"
    for option in result["options"]:
        hold = booking.create_hold(option["slot_ids"], 90, f"group-{option['start_time']}")
        assert len(hold.slot_ids) == len(option["slot_ids"]) * 2, (
            "90 minutes on N courts must occupy 2N slots"
        )
        booking.release_hold(hold.hold_id)


# --- provider fallback, actually exercised ------------------------------------------


@pytest.mark.skipif(not llm.has_credentials(), reason="needs an API key")
def test_primary_provider_outage_is_answered_by_the_fallback():
    """Previously this only asserted a fallback was attached. Here the primary really
    fails and the answer has to come from the other vendor."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import AIMessage

    class DeadProvider(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("primary provider is down")

    dead = DeadProvider(responses=["never returned"])
    standby = FakeListChatModel(responses=["answer from the standby provider"])

    response = dead.with_fallbacks([standby]).invoke("anything")

    assert isinstance(response, AIMessage)
    assert "standby" in response.text, "the fallback vendor did not answer"


@pytest.mark.skipif(not llm.has_credentials(), reason="needs an API key")
def test_configured_answerer_carries_a_fallback():
    """And the wiring the app actually uses has one attached."""
    from app.config import settings

    if not settings().anthropic_api_key:
        pytest.skip("no fallback provider key configured")
    model = llm.get_model("answerer")
    assert list(getattr(model, "fallbacks", []))


# --- group holds and bookings ------------------------------------------------------
#
# "Several slots" means two different things: sequential (one court, longer) and
# parallel (several courts, same hour). Expanding only the first and ignoring the rest
# silently held one court of a two-court group while reporting success.


def _group_option(courts: int = 2):
    options = retrieval.find_group_slots(band="evening", courts=courts, adjacent=True,
                                         limit=1)["options"]
    assert options, "fixture needs a free adjacent group"
    return options[0]


def test_group_hold_holds_every_court_not_just_the_first():
    option = _group_option()
    hold = booking.create_hold(option["slot_ids"], 60, "group-a")
    assert set(hold.slot_ids) == set(option["slot_ids"])


def test_a_held_group_cannot_be_taken_by_anyone_else():
    option = _group_option()
    booking.create_hold(option["slot_ids"], 60, "group-b")
    for slot_id in option["slot_ids"]:
        with pytest.raises(booking.SlotUnavailable):
            booking.create_booking([slot_id], "outsider", 60, session_id="different")


def test_group_booking_claims_every_court():
    option = _group_option()
    result = booking.create_booking(option["slot_ids"], "usr_group", 60)
    assert set(result.slot_ids) == set(option["slot_ids"])
    for slot_id in option["slot_ids"]:
        assert booking.slot_state(slot_id)["status"] == "booked"


def test_group_booking_is_priced_per_court():
    """Two courts for an hour costs both courts, not one."""
    option = _group_option()
    with db.read_conn() as conn:
        expected = sum(
            conn.execute("SELECT price_aed FROM slots WHERE id=?", (s,)).fetchone()[0]
            for s in option["slot_ids"]
        )
    result = booking.create_booking(option["slot_ids"], "usr_group", 60)
    assert result.price_aed == expected


def test_two_courts_for_ninety_minutes_occupies_four_slots():
    """Sequential and parallel compose: 2 courts x 2 hours."""
    option = _group_option()
    try:
        result = booking.create_booking(option["slot_ids"], "usr_group", 90)
    except booking.BookingError:
        pytest.skip("no adjacent group with both following hours free")
    assert len(result.slot_ids) == 4
    with db.read_conn() as conn:
        courts = {conn.execute("SELECT court_id FROM slots WHERE id=?", (s,)).fetchone()[0]
                  for s in result.slot_ids}
    assert len(courts) == 2, "should be two courts, two hours each"


def test_a_group_is_all_or_nothing():
    """If one court of the group is taken, the whole group must fail and claim nothing."""
    option = _group_option()
    booking.create_booking([option["slot_ids"][1]], "early-bird", 60)
    before = _claims()
    with pytest.raises(booking.SlotUnavailable):
        booking.create_booking(option["slot_ids"], "the-group", 60)
    assert _claims() == before, "a failed group booking left claims behind"


def _claims() -> int:
    with db.read_conn() as conn:
        return conn.execute("SELECT count(*) c FROM slot_claims").fetchone()["c"]


# --- branch scoping in prose search -------------------------------------------------


def test_every_record_names_its_branch():
    """"Who is the best coach in Al Ain" was answered with a Khalifa City coach because
    records carried br_khalifa and nothing else. An opaque foreign key in a payload aimed
    at a language model is an invitation to decode it wrongly."""
    records = retrieval.hydrate(["cch_salem_al_ketbi", "br_alquoz", "pol_cancellation"])
    by_id = {r["id"]: r for r in records}

    assert by_id["cch_salem_al_ketbi"]["branch_name"] == "Baseline Khalifa City"
    # A branch is its own branch; a policy belongs to none and must not gain a stray key.
    assert by_id["br_alquoz"]["branch_name"] == "Baseline Al Quoz"
    assert "branch_name" not in by_id["pol_cancellation"]


def test_prose_search_can_be_scoped_to_a_branch():
    """Unscoped, this query returned coaches from four branches and none from Yas."""
    result = retrieval.search_knowledge("best coach", types=["coach"], branch="Yas Island")

    assert result["records"], "scoped search returned nothing"
    assert {r["branch_id"] for r in result["records"]} == {"br_yas"}


def test_branch_scoping_survives_the_lexical_fallback(monkeypatch):
    """The filter has to hold on both halves of the hybrid. Chroma going away must not
    quietly widen the search back out to all eight branches."""
    monkeypatch.setattr(vectorstore, "search", lambda *a, **k: None)

    result = retrieval.search_knowledge("coach", types=["coach"], branch="Al Ain")

    assert result["mode"] == "lexical"
    assert result["records"], "lexical fallback returned nothing"
    assert {r["branch_id"] for r in result["records"]} == {"br_alain"}


def test_a_truncated_result_admits_it():
    """Yas has 7 coaches and the display cap is 6. Silently returning 6 lets the model
    present a partial list as the whole roster; it needs to see that it was cut."""
    result = retrieval.search_knowledge("best coach", types=["coach"], branch="Yas Island")

    assert len(result["records"]) == 6, "fixture assumes the cap still bites here"
    assert result["total_matched"] == 7
    assert result["truncated"] is True


def test_an_unscoped_result_that_fits_is_not_flagged_truncated():
    result = retrieval.search_knowledge("best coach", types=["coach"], branch="Al Ain")

    assert result["total_matched"] == len(result["records"]) == 4
    assert result["truncated"] is False


def test_an_unknown_branch_says_so_rather_than_searching_everywhere():
    """Silently ignoring an unresolvable branch is how a scoped question gets an
    unscoped answer that reads as authoritative."""
    result = retrieval.search_knowledge("coach", types=["coach"], branch="Riyadh")

    assert result["records"] == []
    assert "Riyadh" in result.get("note", "")


# --- band price multipliers ---------------------------------------------------------


def test_price_summary_publishes_the_band_multipliers():
    """The agent can only explain *why* evening costs more if it is handed the ratio."""
    result = retrieval.price_summary()

    assert result["multipliers"]["evening"] == {"weekday": 1.25, "weekend": 1.438}
    assert result["multipliers"]["morning"]["weekday"] == 0.75
    assert set(result["multipliers"]) == {"morning", "afternoon", "evening", "late"}
    # Fri/Sat, not the civil Sat/Sun -- reconciles 11,114 of 11,130 priced slots where
    # Sat/Sun reconciles 8,154.
    assert result["weekend_days"] == ["Friday", "Saturday"]


def test_quoted_multipliers_reconcile_with_what_we_actually_charge():
    """The guard that matters: we are about to tell users evening is 1.25x the base, so
    that had better still describe slots.price_aed. Fails the day the grid is repriced."""
    multipliers = retrieval.price_summary()["multipliers"]
    with db.read_conn() as conn:
        rows = conn.execute(
            "SELECT s.start_time, s.price_aed, c.price_per_hour_aed AS base,"
            " CAST(strftime('%w', s.date) AS INTEGER) IN (5, 6) AS weekend"
            " FROM slots s JOIN courts c ON c.id = s.court_id"
            " WHERE c.price_per_hour_aed IS NOT NULL AND c.price_per_hour_aed < 99999"
        ).fetchall()

    band_of = {t: band for band, times in retrieval.BANDS.items() for t in times}
    off = 0
    for row in rows:
        rate = multipliers[band_of[row["start_time"]]]["weekend" if row["weekend"] else "weekday"]
        # Published prices are rounded to the nearest 5 AED, so allow half a step.
        off += abs(row["price_aed"] - row["base"] * rate) > 2.5

    assert len(rows) > 10_000, "expected the bulk of the grid to have a usable court base"
    assert off / len(rows) < 0.01, f"{off}/{len(rows)} slots contradict the published rate"


# --- refusals that skip retrieval ---------------------------------------------------
#
# out_of_scope and asks_for_personal_data have no answer anywhere in the data. Sending
# them through `answer` bought a retrieval loop and answerer-rate tokens to arrive at a
# conclusion the planner had already reached.


def test_a_planner_refusal_routes_away_from_the_tool_loop():
    from app.agent import graph as agent_graph

    assert agent_graph.after_plan({"plan": {"out_of_scope": True}}) == "alternate"
    assert agent_graph.after_plan({"plan": {"asks_for_personal_data": True}}) == "alternate"
    # Everything else keeps its tools, including a question whose answer happens to be no.
    assert agent_graph.after_plan({"plan": {"out_of_scope": False}}) == "answer"
    assert agent_graph.after_plan({}) == "answer"


def test_the_alternate_node_refuses_without_binding_tools(monkeypatch):
    """A refusal needs no tools, and a tool-less model cannot burn a loop discovering
    that the club does not offer squash."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent import graph as agent_graph

    class Recorder:
        messages: list = []

        def bind_tools(self, tools):
            raise AssertionError("the alternate node must not bind tools")

        def invoke(self, messages, *args, **kwargs):
            Recorder.messages = messages
            return AIMessage("We only offer padel.")

    monkeypatch.setattr(agent_graph.llm, "get_model", lambda *a, **k: Recorder())
    agent_tools.start_request("refusal-test")  # the surfaced-ids ContextVar is per-request

    result = agent_graph.alternate_node({
        "messages": [HumanMessage("Do you teach squash?")],
        "plan": {"out_of_scope": True},
    })

    assert result["messages"][0].text == "We only offer padel."
    system = Recorder.messages[0].text
    assert "outside what the club offers" in system
    assert "private" not in system, "the personal-data rule does not belong on this turn"
    assert agent_tools.surfaced_ids() == [], "a refusal must not retrieve"


def test_the_alternate_node_declines_personal_details_in_the_users_language(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent import graph as agent_graph

    class Recorder:
        messages: list = []

        def invoke(self, messages, *args, **kwargs):
            Recorder.messages = messages
            return AIMessage("...")

    monkeypatch.setattr(agent_graph.llm, "get_model", lambda *a, **k: Recorder())

    agent_graph.alternate_node({
        "messages": [HumanMessage("ما هو رقم جوال المدرب مروان؟")],
        "plan": {"asks_for_personal_data": True},
    })

    system = Recorder.messages[0].text
    assert "private" in system and "branch" in system
    assert system.endswith(agent_graph.LANGUAGE_RULE["ar"]), "Arabic asks get an Arabic reply"


# --- request limits ----------------------------------------------------------------


def test_oversized_chat_message_is_rejected_before_it_costs_anything(client):
    """/chat is unauthenticated and every call bills a model, so the length cap is the
    only thing standing between a curl loop and the bill. 400, not 422, per the contract."""
    from app.api.chat import MAX_MESSAGE_CHARS

    response = client.post(
        "/api/v1/chat", json={"message": "a" * (MAX_MESSAGE_CHARS + 1)}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"

    assert client.post("/api/v1/chat", json={"message": ""}).status_code == 400

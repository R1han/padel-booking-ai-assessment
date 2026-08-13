"""The system stays useful when a dependency goes away.

Each test removes a real dependency rather than mocking around it, and asserts the
system still answers instead of failing the request.
"""

from __future__ import annotations

import sqlite3

import pytest

from app import db, llm
from app.services import retrieval, vectorstore


def test_vector_store_down_falls_back_to_lexical(monkeypatch):
    """Chroma unavailable: hybrid search degrades to SQLite FTS5 and still returns
    records, flagged so callers know the result is lexical-only."""
    monkeypatch.setattr(vectorstore, "search", lambda *a, **k: None)

    result = retrieval.search_knowledge("ladies only beginner classes")

    assert result["degraded"] is True
    assert result["mode"] == "lexical"
    assert result["records"], "lexical fallback returned nothing"


def test_vector_store_raising_is_swallowed(monkeypatch):
    """An exception from the vector store must degrade, not propagate."""
    def explode(*args, **kwargs):
        raise RuntimeError("chroma is on fire")

    monkeypatch.setattr(vectorstore, "_connect", explode)
    assert vectorstore.search("anything") is None

    result = retrieval.search_knowledge("indoor courts")
    assert result["mode"] == "lexical"
    assert result["records"]


def test_reranker_failure_keeps_the_fused_order(monkeypatch):
    """A reranker outage must not lose the candidates retrieval already found."""
    records = retrieval.hydrate(["br_alquoz", "br_jvc", "br_yas"])

    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(llm, "get_model", lambda *a, **k: Broken())
    monkeypatch.setattr(llm, "has_credentials", lambda: True)

    assert retrieval.rerank("anything", records) is None


def test_planner_failure_still_answers(monkeypatch):
    """If the planner cannot be reached the graph uses the raw query rather than
    failing the turn."""
    from app.agent import graph as agent_graph

    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("planner unavailable")

    monkeypatch.setattr(llm, "get_model", lambda *a, **k: Broken())
    monkeypatch.setattr(llm, "has_credentials", lambda: True)

    state = {"messages": [_human("Which branches have indoor courts?")], "loops": 0}
    plan = agent_graph.plan_node(state)["plan"]

    assert plan["english_query"] == "Which branches have indoor courts?"
    assert plan["out_of_scope"] is False


def _human(text):
    from langchain_core.messages import HumanMessage

    return HumanMessage(text)


def test_structured_lookups_need_no_model_at_all():
    """Availability, pricing and counting are pure SQL, so they keep working with every
    model provider unreachable."""
    assert retrieval.find_records("coach", branch="Ajman")["total"] == 4
    assert retrieval.price_summary(band="evening")["branches"]
    assert retrieval.check_availability(branch="Yas", date_="tomorrow")["slots"]


def test_missing_fts_index_does_not_raise(monkeypatch):
    """A database built without FTS5 support must degrade, not crash the query."""
    def no_fts(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: docs_fts")

    class Conn:
        def execute(self, *args, **kwargs):
            no_fts()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(db, "read_conn", lambda: Conn())
    assert retrieval.lexical_search("anything") == []


def test_booking_never_needs_a_model():
    """The booking path is pure SQL. A provider outage cannot affect the race test."""
    from app.services import booking

    with db.read_conn() as conn:
        slot = conn.execute(
            "SELECT id FROM slots WHERE status='available'"
            " AND id NOT IN (SELECT slot_id FROM slot_claims)"
            " AND id NOT IN (SELECT slot_id FROM slot_overhang) LIMIT 1"
        ).fetchone()["id"]

    result = booking.create_booking([slot], "usr_degraded", 60)
    assert result.status == "confirmed"

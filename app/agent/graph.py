"""The conversational agent.

    plan -> answer <-> tools -> END

`plan` is one cheap structured call that normalises the query to English, resolves
cross-turn references against what was shown last turn, and flags obvious out-of-scope
asks. `answer` is the grounded generation loop with tools bound.

The split exists for two reasons beyond tidiness: it gives LangSmith a per-node
breakdown of latency, tokens and cost, and it lets us skip work -- an out-of-scope
question is refused without ever touching retrieval or the expensive model.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app import llm
from app.agent import tools as agent_tools
from app.config import settings
from app.services import retrieval

log = logging.getLogger("padel.graph")

MAX_TOOL_LOOPS = 6

PLANNER_PROMPT = """You prepare a user's message for a padel club assistant.

Today is {today}. Bookable dates run {window_start} to {window_end}.
The club has eight branches, all in the UAE: Dubai (Al Quoz, JVC), Abu Dhabi (Yas Island,
Khalifa City, Al Ain), Sharjah (Al Majaz), Ajman, and Ras Al Khaimah.
It offers padel only.

Return JSON with exactly these keys:
  "language": "en", "ar" or "mixed"
  "english_query": the message rewritten in English, self-contained. Resolve references to
      earlier turns ("the second one", "same time at Yas", "book it") into explicit terms
      using the context below. Keep proper nouns as written.
  "out_of_scope": true only if the request cannot possibly be about this club -- another
      sport, another country or city we have no branch in, or an unrelated topic.
  "asks_for_personal_data": true if it requests a staff member's private contact details,
      home address, salary or similar.
  "referenced_ids": ids from the context below that the message points at, or [].

Context from the previous turn (may be empty):
{context}

User message:
{message}"""

ANSWER_PROMPT = """You are the assistant for Baseline Padel, a padel club with eight
branches across the UAE.

Today's date is {today}. Bookable dates run {window_start} to {window_end}.

GROUNDING -- this matters more than being helpful:
* Answer only from what the tools return. If the tools do not support a claim, do not
  make it.
* Never invent a branch, court, coach, class, package, price, or availability. If you
  cannot find something, say plainly that you do not have it.
* Prices, availability and schedules must come from a tool call in this conversation.
  Do not reuse a number from memory or infer one.
* If a court code matches several branches, ask which branch. Do not pick one.
* A coach with no published schedule for a date has unknown availability. Say that,
  rather than implying they are free.
* A package whose validity window has ended is expired, whatever its status field says.
* Decline requests for staff personal contact details. The club's public branch phone
  numbers are fine to share.
* If the question is outside what this club offers -- another sport, another country,
  a city with no branch -- say so directly and briefly.
* Be complete. When a tool returns `counts_by_branch` or `total`, report every entry and
  use those numbers verbatim. Do not drop an entry because its count is small, and never
  count by tallying a truncated record list.

BOOKING:
* Use slot ids exactly as check_availability returned them. Never construct, guess or
  reassemble a slot id from a court code and a time -- if you do not have the id, call
  check_availability again.
* Confirm the exact slot, date, time, duration and price with the user before booking.
* Call hold_slots when you propose a specific slot, so it is not taken while they decide.
* Call book_court only after they explicitly agree.
* Report the returned booking_id exactly. If a booking fails, say why in plain language.

STYLE:
* Be brief and concrete. Lead with the answer, then the detail.
* Quote prices in AED and use 24-hour times as they appear in the data.
* Keep branch, coach and court names exactly as they appear in the data, even when the
  rest of your reply is in Arabic.
"""

# Stated separately and last so it cannot be diluted by the rest of the prompt. Without
# an explicit instruction the model answered English questions in Arabic.
LANGUAGE_RULE = {
    "en": "Reply in English. The user wrote in English.",
    "ar": "Reply in Arabic. The user wrote in Arabic.",
    "mixed": "Reply in the language the user's own words are mostly written in.",
}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    context: str
    plan: dict[str, Any]
    refused: bool
    loops: int


def _window() -> dict[str, str]:
    from app import db

    with db.read_conn() as conn:
        row = conn.execute("SELECT MIN(date) a, MAX(date) b FROM slots").fetchone()
    return {"window_start": row["a"], "window_end": row["b"]}


def _last_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            return message.text if isinstance(message.text, str) else str(message.content)
    return ""


def plan_node(state: AgentState) -> dict:
    """Normalise the query and resolve cross-turn references before any retrieval."""
    message = _last_user_text(state)
    plan: dict[str, Any] = {
        "language": "en", "english_query": message,
        "out_of_scope": False, "asks_for_personal_data": False, "referenced_ids": [],
    }
    if not llm.has_credentials():
        return {"plan": plan}

    prompt = PLANNER_PROMPT.format(
        today=retrieval.today().isoformat(),
        context=state.get("context") or "(none)",
        message=message,
        **_window(),
    )
    try:
        response = llm.get_model("planner").invoke(prompt)
        llm.record_response(response, llm.model_spec("planner"), step="plan")
        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
        plan.update({k: v for k, v in parsed.items() if k in plan})
    except Exception as exc:  # noqa: BLE001 - a planner failure degrades, never fails
        log.warning("planner unavailable, using the raw query: %s", exc)
    return {"plan": plan}


def answer_node(state: AgentState) -> dict:
    plan = state.get("plan", {})
    model = llm.get_model("answerer").bind_tools(agent_tools.ALL_TOOLS)

    system = ANSWER_PROMPT.format(today=retrieval.today().isoformat(), **_window())
    if plan.get("english_query") and plan["english_query"] != _last_user_text(state):
        system += (
            f"\nSearch using this English reading of the request: "
            f"{plan['english_query']}\n"
        )
    if plan.get("out_of_scope"):
        system += "\nThis looks outside what the club offers. Verify, then say so plainly.\n"
    if plan.get("asks_for_personal_data"):
        system += "\nThis asks for private staff details. Decline and offer the branch line.\n"
    system += "\n" + LANGUAGE_RULE.get(plan.get("language", "en"), LANGUAGE_RULE["en"])

    messages = [SystemMessage(system), *state["messages"]]
    if state.get("loops", 0) >= MAX_TOOL_LOOPS:
        messages.append(SystemMessage(
            "Tool budget spent. Answer from what you already have, or say you could not "
            "find it. Do not call further tools."
        ))
        model = llm.get_model("answerer")

    response = model.invoke(messages)
    llm.record_response(response, llm.model_spec("answerer"), step="answer")
    return {"messages": [response], "loops": state.get("loops", 0) + 1}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and state.get("loops", 0) < MAX_TOOL_LOOPS:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tools", ToolNode(agent_tools.ALL_TOOLS))
    graph.set_entry_point("plan")
    graph.add_edge("plan", "answer")
    graph.add_conditional_edges("answer", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "answer")
    return graph.compile()


_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# --- refusal detection ------------------------------------------------------------

# Declining to answer. Kept narrow: these say "I do not hold this", not "the answer is no".
REFUSAL_MARKERS = (
    "i don't have", "i do not have", "i don't hold", "no information",
    "cannot share", "can't share", "not able to share", "cannot provide",
    "can't provide", "i can't help", "cannot help", "no record",
    "don't provide", "do not provide", "outside the",
    "لا أستطيع", "لا يمكنني", "ليس لدي معلومات", "لا نملك معلومات", "لا توجد معلومات",
)

# Reporting that nothing is free IS an answer, not a refusal.
ANSWERED_NEGATIVES = (
    "no available", "no free", "not available", "fully booked", "is booked",
    "no slots", "no courts available", "already booked",
    "غير متاح", "محجوز", "لا يوجد ملاعب متاحة", "لا توجد ملاعب",
)

# A reply that opens affirmatively has answered, whatever caveats follow.
AFFIRMATIVE_OPENERS = ("yes", "نعم", "sure", "certainly", "أجل")

REFUSAL_WINDOW = 160


def _opening(answer: str) -> str:
    """The first sentence. A genuine refusal declines there; a caveat further down
    ('...but we hold no information about pools') belongs to an answer."""
    for separator in ("\n", ". ", "؟ ", "! ", "? "):
        answer = answer.split(separator)[0]
    return answer.strip().lower()


def looks_like_refusal(answer: str, surfaced: list[str], plan: dict) -> bool:
    """The eval contract scores `refused` in both directions, so this has to be honest
    rather than optimistic.

    Two distinctions cost us accuracy before they were made explicit:

    * "we hold no such information" is a refusal; "there is nothing free at that time"
      is an answer. Availability negatives are therefore checked first.
    * "no, we don't do instalments, billing is monthly" is a grounded negative answer,
      not a refusal. So markers are only honoured near the start of the reply, and
      questions of scope are decided by the planner rather than by string matching.
    """
    if plan.get("out_of_scope") or plan.get("asks_for_personal_data"):
        return True

    lowered = (answer or "").lower()
    opening = _opening(answer or "")
    if opening.startswith(AFFIRMATIVE_OPENERS):
        return False
    if any(marker in lowered for marker in ANSWERED_NEGATIVES):
        return False
    if any(marker in opening for marker in REFUSAL_MARKERS):
        return True
    # Nothing retrieved and nothing substantive said.
    return not surfaced and len(answer) < REFUSAL_WINDOW


# --- one-shot entry point ---------------------------------------------------------


async def run_query(
    message: str, session_id: str = "eval", history: list | None = None, context: str = ""
) -> dict:
    """Run a full turn and return everything the eval contract needs.

    Shared by the eval harness and the chat endpoint so the numbers we report are the
    numbers the app actually produces.
    """
    import time

    from langchain_core.messages import HumanMessage

    usage = llm.start_usage()
    surfaced = agent_tools.start_request(session_id)
    started = time.perf_counter()

    state: AgentState = {
        "messages": [*(history or []), HumanMessage(message)],
        "session_id": session_id,
        "context": context,
        "loops": 0,
    }
    result = await graph().ainvoke(state)
    latency_ms = round((time.perf_counter() - started) * 1000)

    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.text if isinstance(msg.text, str) else str(msg.content)
            break

    plan = result.get("plan", {})
    return {
        "answer": answer,
        "retrieved_ids": list(surfaced),
        "refused": looks_like_refusal(answer, surfaced, plan),
        "latency_ms": latency_ms,
        "cost_usd": usage.cost_usd,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "steps": usage.calls,
        "plan": plan,
        "messages": result["messages"],
    }

"""Streaming chat over Server-Sent Events.

History lives in a process-local dict and dies with the process: the brief asks for no
auth, no accounts and no persistence beyond the active session.

The last turn's surfaced records are kept alongside it so the planner can resolve
"the second one" or "same time at Yas" into concrete ids.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app import llm
from app.agent import graph as agent_graph
from app.agent import tools as agent_tools

log = logging.getLogger("padel.chat")

router = APIRouter(prefix="/api/v1", tags=["chat"])

MAX_SESSIONS = 200
MAX_TURNS = 12


class Session:
    def __init__(self) -> None:
        self.messages: list = []
        self.context: str = ""


_sessions: "OrderedDict[str, Session]" = OrderedDict()


def session_for(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session()
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    _sessions.move_to_end(session_id)
    return _sessions[session_id]


def summarise_context(records: list[dict], answer: str) -> str:
    """A compact note of what was just shown, so the next turn can point back at it."""
    lines = []
    for i, record in enumerate(records[:8], 1):
        label = record.get("name") or record.get("title") or record.get("id")
        extra = ""
        if record.get("start_time"):
            extra = f" {record.get('date')} {record['start_time']} {record.get('price_aed')} AED"
        lines.append(f"{i}. {record['id']} - {label}{extra}")
    shown = "\n".join(lines) or "(nothing specific)"
    return f"Assistant's last reply:\n{answer[:600]}\n\nRecords shown, in order:\n{shown}"


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def _stream(req: ChatRequest) -> AsyncIterator[str]:
    session = session_for(req.session_id)
    usage = llm.start_usage()
    surfaced = agent_tools.start_request(req.session_id)
    started = time.perf_counter()
    first_token_ms: float | None = None

    state = {
        "messages": [*session.messages, HumanMessage(req.message)],
        "session_id": req.session_id,
        "context": session.context,
        "loops": 0,
    }

    answer_parts: list[str] = []
    try:
        async for chunk, meta in agent_graph.graph().astream(state, stream_mode="messages"):
            node = meta.get("langgraph_node")
            if node == "tools":
                continue
            if node == "plan":
                continue  # internal reasoning, never shown to the user
            text = getattr(chunk, "text", None)
            text = text if isinstance(text, str) else ""
            if text:
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000)
                    yield _sse("start", {"ttft_ms": first_token_ms})
                answer_parts.append(text)
                yield _sse("token", {"text": text})
            for call in getattr(chunk, "tool_calls", None) or []:
                if call.get("name"):
                    yield _sse("tool", {"name": call["name"]})
    except Exception as exc:  # noqa: BLE001 - the stream must always close cleanly
        log.exception("chat stream failed")
        yield _sse("error", {"message": f"Something went wrong: {exc}"})

    answer = "".join(answer_parts)
    records = agent_tools.surfaced_records()

    session.messages = [*state["messages"], AIMessage(answer)][-MAX_TURNS * 2 :]
    session.context = summarise_context(records, answer)

    yield _sse("done", {
        "retrieved_ids": list(surfaced),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "ttft_ms": first_token_ms,
        "cost_usd": usage.cost_usd,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "steps": usage.calls,
        "slots": [r for r in records if r.get("start_time")][:8],
    })


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/chat/{session_id}")
def reset_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"reset": session_id}

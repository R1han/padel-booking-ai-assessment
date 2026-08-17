"""LLM claim extraction over the official Anthropic SDK.

One call per catalog record. The response schema is enforced by the API via
`messages.parse`, so there is no JSON to parse and no retry-on-parse loop.

Without ANTHROPIC_API_KEY, get_llm() returns None. There is no offline
fallback: run_pipeline.py treats a missing key as a hard failure rather than
running with less evidence.
"""
from __future__ import annotations

import logging
import os

import anthropic
from pydantic import BaseModel

from .claims import CLAIM_MODELS

log = logging.getLogger("pipeline.llm")

MODEL = "claude-opus-5"
MAX_TOKENS = 16000  # thinking is on by default on Opus 5 and shares this budget

SYSTEM = """You extract structured claims from padel-club catalogue copy.

Rules, both absolute:

1. Return a field only if the text EXPLICITLY states it. If the text does not
   mention a field, leave `stated` false and `value` null. Do not infer, do not
   guess, and do not carry a value over from a neighbouring field. A plausible
   answer you cannot point to in the text is a wrong answer.

2. Every field you mark `stated` must carry a VERBATIM quote from the text in
   `evidence` — copied character for character, not paraphrased.

Note the difference between a field the text is silent about (`stated` false)
and a field the text explicitly says is unbounded, such as a class open to all
ages above a minimum (`stated` true, `value` null).

`confidence` is how strongly the quote supports the value, 0 to 1."""

ENTITY_NOUN = {
    "courts": "padel court",
    "coaches": "padel coach",
    "classes": "padel class or course",
    "packages": "membership or session package",
    "branches": "padel club branch",
}


def build_prompt(entity: str, text: str) -> str:
    return (
        f"Extract claims about this {ENTITY_NOUN[entity]} from the text below.\n"
        f"Any field not explicitly stated in the text must be left unstated.\n\n"
        f"---\n{text}\n---"
    )


def verify_evidence(obj: BaseModel, text: str) -> BaseModel:
    """Downgrade any stated claim whose evidence is not a verbatim quote.

    An unquotable claim is an invented one. Cheap, deterministic, and it costs
    nothing to run on every field.
    """
    haystack = " ".join(text.split()).lower()
    for attr in obj.__class__.model_fields:
        claim = getattr(obj, attr)
        if not claim.stated:
            continue
        quote = " ".join((claim.evidence or "").split()).lower()
        if not quote or quote not in haystack:
            claim.stated = False
            claim.confidence = 0.0
            claim.evidence = f"discarded: quote not found in source text ({claim.evidence!r})"
    return obj


class Extractor:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    def extract(self, entity: str, record: dict, text: str) -> BaseModel | None:
        """Return the entity's claim object, or None if the call failed."""
        schema = CLAIM_MODELS[entity]
        try:
            resp = self.client.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(entity, text)}],
                output_format=schema,
            )
        except anthropic.APIError as e:
            log.warning("extraction failed for %s/%s: %s", entity, record.get("id"), e)
            return None
        if resp.stop_reason == "refusal":
            log.warning("extraction refused for %s/%s", entity, record.get("id"))
            return None
        return verify_evidence(resp.parsed_output, text)


def get_llm() -> Extractor | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return Extractor(anthropic.Anthropic())

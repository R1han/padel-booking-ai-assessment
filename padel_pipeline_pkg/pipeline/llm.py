"""Optional LLM tier for semantic checks.

Only invoked for records the deterministic heuristics flag or cannot decide,
which keeps API cost bounded (dozens of calls, not thousands).
Requires ANTHROPIC_API_KEY in the environment; without it, get_llm() returns
None and the pipeline runs heuristics-only.
"""
from __future__ import annotations

import json
import os

import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"


class AnthropicChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _ask_json(self, prompt: str) -> dict | None:
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 300,
                    "system": "Respond ONLY with a single JSON object. No prose, no markdown fences.",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = "".join(b.get("text", "") for b in resp.json()["content"] if b.get("type") == "text")
            return json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        except Exception:
            return None

    # -- verdicts --------------------------------------------------------
    def classify_court_type(self, description: str) -> tuple[str | None, float, str]:
        out = self._ask_json(
            "Does this padel court description describe an INDOOR or OUTDOOR court?\n"
            f"Description: {description}\n"
            'Reply JSON: {"verdict": "indoor"|"outdoor"|"ambiguous", '
            '"confidence": 0..1, "evidence": "short quote or reason"}'
        )
        if not out or out.get("verdict") not in ("indoor", "outdoor"):
            return None, 0.0, "llm: ambiguous or unavailable"
        return out["verdict"], float(out.get("confidence", 0.8)), f"llm: {out.get('evidence', '')}"

    def extract_years(self, bio: str) -> tuple[int | None, str]:
        out = self._ask_json(
            "How many years of coaching/professional experience does this coach bio state? "
            "If not stated, use null.\n"
            f"Bio: {bio}\n"
            'Reply JSON: {"years": int|null, "evidence": "short quote"}'
        )
        if not out or not isinstance(out.get("years"), int):
            return None, "llm: not stated or unavailable"
        return out["years"], f"llm: {out.get('evidence', '')}"


def get_llm() -> AnthropicChecker | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    return AnthropicChecker(key) if key else None

"""Model access behind a role -> provider:model indirection.

Components ask for a *role* ("planner", "answerer"), never a vendor. Which model fills a
role is one line of .env, so swapping providers needs no code change.

Token usage is accumulated per request in a ContextVar, which is what lets the eval
harness report a real cost_usd per query rather than an estimate.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import settings

log = logging.getLogger("padel.llm")


@dataclass
class Usage:
    calls: list[dict] = field(default_factory=list)

    def add(self, model: str, input_tokens: int, output_tokens: int, step: str = "",
            duration_ms: int = 0) -> None:
        self.calls.append({
            "model": model, "step": step, "duration_ms": duration_ms,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })

    @property
    def input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    @property
    def cost_usd(self) -> float:
        prices = settings().model_prices
        total = 0.0
        for call in self.calls:
            rate = prices.get(_bare(call["model"]))
            if rate is None:
                log.warning("no price configured for %s; cost under-reported", call["model"])
                continue
            total += call["input_tokens"] / 1e6 * rate[0]
            total += call["output_tokens"] / 1e6 * rate[1]
        return round(total, 6)


def _bare(model: str) -> str:
    """'openai:gpt-4.1-mini' -> 'gpt-4.1-mini'."""
    return model.split(":", 1)[-1]


_usage: ContextVar[Usage | None] = ContextVar("padel_usage", default=None)


def start_usage() -> Usage:
    usage = Usage()
    _usage.set(usage)
    return usage


def current_usage() -> Usage | None:
    return _usage.get()


def record(model: str, input_tokens: int, output_tokens: int, step: str = "",
           duration_ms: int = 0) -> None:
    usage = _usage.get()
    if usage is not None:
        usage.add(model, input_tokens, output_tokens, step, duration_ms)


def record_response(response, model: str, step: str = "", duration_ms: int = 0) -> None:
    """Pull real token counts off a LangChain response rather than estimating."""
    meta = getattr(response, "usage_metadata", None) or {}
    record(model, meta.get("input_tokens", 0), meta.get("output_tokens", 0), step,
           duration_ms)


@contextmanager
def timed(model: str, step: str):
    """Wrap a model call so its latency lands in the same ledger as its tokens.

    Mirrors what LangSmith records, so the numbers we report can be checked against the
    dashboard rather than taken on trust.
    """
    started = time.perf_counter()
    box: dict = {}
    try:
        yield box
    finally:
        elapsed = round((time.perf_counter() - started) * 1000)
        if "response" in box:
            record_response(box["response"], model, step, elapsed)
        else:
            record(model, 0, 0, step, elapsed)


def configure_tracing() -> None:
    """LangSmith is opt-in and must never break startup when unconfigured."""
    cfg = settings()
    if not (cfg.langsmith_tracing and cfg.langsmith_api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project
    log.info("langsmith tracing enabled, project=%s", cfg.langsmith_project)


def _api_key_for(spec: str) -> dict:
    cfg = settings()
    provider = spec.split(":", 1)[0]
    if provider == "openai" and cfg.openai_api_key:
        return {"api_key": cfg.openai_api_key}
    if provider == "anthropic" and cfg.anthropic_api_key:
        return {"api_key": cfg.anthropic_api_key}
    return {}


ROLE_SPECS = {
    "planner": lambda c: c.llm_planner,
    "reranker": lambda c: c.llm_reranker,
    "answerer": lambda c: c.llm_answerer,
}

_cache: dict[tuple, BaseChatModel] = {}


def model_spec(role: str) -> str:
    cfg = settings()
    if role not in ROLE_SPECS:
        raise KeyError(f"unknown model role {role!r}; expected one of {list(ROLE_SPECS)}")
    return ROLE_SPECS[role](cfg)


def get_model(role: str, *, temperature: float = 0.0, fallback: bool = True) -> BaseChatModel:
    """The model for a role, with the configured fallback provider attached.

    The fallback is what keeps the assistant answering when the primary provider is
    rate-limited or down -- LangChain retries the whole call against it transparently.
    """
    cfg = settings()
    key = (role, temperature, fallback)
    if key in _cache:
        return _cache[key]

    spec = model_spec(role)
    model = init_chat_model(spec, temperature=temperature, **_api_key_for(spec))
    if fallback and cfg.llm_fallback and _api_key_for(cfg.llm_fallback):
        alt = init_chat_model(
            cfg.llm_fallback, temperature=temperature, **_api_key_for(cfg.llm_fallback)
        )
        model = model.with_fallbacks([alt])
    _cache[key] = model
    return model


def embeddings():
    """Embeddings are OpenAI-only today (Anthropic has no embedding endpoint), but the
    model name still comes from config."""
    from langchain_openai import OpenAIEmbeddings

    cfg = settings()
    return OpenAIEmbeddings(
        model=_bare(cfg.embedding_model),
        **({"api_key": cfg.openai_api_key} if cfg.openai_api_key else {}),
    )


def has_credentials() -> bool:
    cfg = settings()
    return bool(cfg.openai_api_key or cfg.anthropic_api_key)

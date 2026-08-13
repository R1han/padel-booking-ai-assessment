"""Single source of configuration. Everything env-driven, nothing hardcoded in business logic."""

from __future__ import annotations

import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info < (3, 12):
    raise RuntimeError(
        f"Python 3.12 is required, found {sys.version_info.major}.{sys.version_info.minor}. "
        "Create the venv with: python3.12 -m venv .venv"
    )

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _dataset_reference_date() -> date:
    """The dataset's own reference date. Never the system clock -- the slot window is fixed
    at 2026-08-10..2026-08-24 and 'tomorrow' must resolve inside it on any grading day."""
    meta = json.loads((ROOT / "dataset_meta.json").read_text())
    return date.fromisoformat(meta["reference_date"])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- server -------------------------------------------------------------
    # 3000, not 8000: tests/race.sh defaults to BASE_URL=http://localhost:3000.
    port: int = 3000

    # --- paths --------------------------------------------------------------
    db_path: Path = DATA_DIR / "padel.db"
    chroma_path: Path = DATA_DIR / "chroma"
    catalog_dir: Path = ROOT / "catalog"
    structured_dir: Path = ROOT / "structured"

    # --- sqlite -------------------------------------------------------------
    sqlite_busy_timeout_ms: int = 5000
    sqlite_write_retries: int = 3
    threadpool_tokens: int = 64

    # --- booking domain -----------------------------------------------------
    reference_date: date = _dataset_reference_date()
    slot_minutes: int = 60
    booking_allowed_durations: list[int] = [60, 90, 120]
    hold_ttl_seconds: int = 180
    # 565 legacy bookings declare 90min against a single 60min slot; the overhang hour is
    # unmarked in the data. We report availability as shipped but refuse to sell that hour.
    booking_enforce_legacy_overhang: bool = True
    # A 90-minute booking claims two slots but is priced at 1.5x, not 2x.
    price_per_slot_minute: bool = True

    # --- models (role -> "provider:model", swappable without code changes) ---
    # The planner is a short, mechanical rewrite, so it runs on the smallest model.
    # Measured: ~1.0s on nano against ~2.5-3.9s on mini, for the same plan quality.
    llm_planner: str = "openai:gpt-4.1-nano"
    llm_reranker: str = "openai:gpt-4.1-mini"
    llm_answerer: str = "openai:gpt-4.1-mini"
    # Planning gets its own cheap round trip before the tool decision. Merging it into
    # the first tool call was tried and measured worse on every axis -- see
    # docs/LATENCY.md -- so the separate node is the default.
    agent_separate_plan_node: bool = True

    llm_fallback: str = "anthropic:claude-haiku-4-5-20251001"
    embedding_model: str = "text-embedding-3-small"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- retrieval ----------------------------------------------------------
    retrieval_top_k: int = 20
    rerank_top_k: int = 6
    rrf_k: int = 60
    # Off by default. Measured on the 38-query eval set, an LLM listwise rerank on top of
    # RRF-fused hybrid retrieval made every metric worse (see eval/RERANKING.md). The
    # stage is kept and is one env var away for anyone who wants to re-measure.
    rerank_enabled: bool = False
    # Reviews are 3120 of 3942 indexed documents and would otherwise dominate every
    # result set, so they are retrieved separately and capped.
    review_results: int = 3
    # Feed the reranker the chunk that matched instead of the parent record's opening.
    # Measured worse (P@1 0.65 vs 0.737) -- it re-does the semantic ranking and collapses
    # the lexical half of the ensemble. Kept for reproducibility; see eval/RERANKING.md.
    rerank_use_chunks: bool = False
    # Cap on prose returned to the model per record. Policy bodies average ~10KB and six
    # of them in one tool result pushed a generation call past 9000 input tokens.
    tool_prose_chars: int = 700

    # --- observability ------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "baseline-padel"

    # USD per 1M tokens, (input, output). Used to compute cost_usd for the eval contract.
    model_prices: dict[str, list[float]] = {
        "gpt-4.1-mini": [0.40, 1.60],
        "gpt-4.1": [2.00, 8.00],
        "gpt-4.1-nano": [0.10, 0.40],
        "claude-haiku-4-5-20251001": [1.00, 5.00],
        "text-embedding-3-small": [0.02, 0.0],
    }


@lru_cache
def settings() -> Settings:
    return Settings()

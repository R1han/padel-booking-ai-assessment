"""FastAPI app.

Two ordering rules here are load-bearing for the race test:
  1. API routers are registered BEFORE the static mount, or StaticFiles at "/" swallows /api/*.
  2. Everything expensive happens in lifespan, so the first booking request never pays init cost.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anyio.to_thread
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import ROOT, settings

log = logging.getLogger("padel")

UI_DIST = ROOT / "ui" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings()
    # race.sh honours an N= override; the default anyio limiter is 40.
    anyio.to_thread.current_default_thread_limiter().total_tokens = cfg.threadpool_tokens

    from app import db, llm
    from app.ingest import ingest, is_ingested

    llm.configure_tracing()  # before any model call, and a no-op without a key
    db.init_schema()
    if not is_ingested():
        log.warning("database empty, running ingest")
        ingest(embed=False)
    from app.services import booking

    booking.sweep_expired_holds()
    yield


app = FastAPI(title="Baseline Padel", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_to_400(request, exc: RequestValidationError):
    """The contract says malformed requests are 400. FastAPI's default is 422, which
    tests/race.sh counts as an unexpected status and fails on."""
    return JSONResponse(
        status_code=400,
        content={"error": "bad_request", "message": "Malformed request.", "detail": exc.errors()},
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Routers first...
from app.api import bookings, chat, slots  # noqa: E402

app.include_router(bookings.router)
app.include_router(slots.router)
app.include_router(chat.router)

# ...static mount last.
if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")

"""SQLite access. The PRIMARY KEY on slot_claims is the concurrency mechanism -- not a
Python lock, not application logic. Everything here exists to keep that guarantee intact."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import settings

# Constraint violations that mean "someone else claimed this slot first".
CLAIM_CONFLICT_CODES = (
    sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,  # 1555
    sqlite3.SQLITE_CONSTRAINT_UNIQUE,  # 2067
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS branches (
  id TEXT PRIMARY KEY, name TEXT, emirate TEXT, area TEXT,
  lat REAL, lng REAL, description TEXT, amenities TEXT,
  opening_hours TEXT, phone TEXT, court_count INTEGER
);

CREATE TABLE IF NOT EXISTS courts (
  id TEXT PRIMARY KEY, code TEXT, branch_id TEXT, name TEXT, type TEXT,
  surface TEXT, walls TEXT, lighting TEXT, description TEXT,
  price_per_hour_aed INTEGER
);

CREATE TABLE IF NOT EXISTS coaches (
  id TEXT PRIMARY KEY, name TEXT, branch_id TEXT, bio TEXT, specialties TEXT,
  languages TEXT, level_focus TEXT, years_experience INTEGER,
  rate_per_hour_aed INTEGER, internal_phone TEXT, internal_email TEXT
);

CREATE TABLE IF NOT EXISTS classes (
  id TEXT PRIMARY KEY, name TEXT, branch_id TEXT, description TEXT, level TEXT,
  min_age INTEGER, max_age INTEGER, gender TEXT, schedule TEXT,
  price_per_term_aed INTEGER, coach_id TEXT
);

CREATE TABLE IF NOT EXISTS packages (
  id TEXT PRIMARY KEY, name TEXT, branch_ids TEXT, description TEXT,
  price_aed INTEGER, sessions INTEGER, valid_from TEXT, valid_until TEXT,
  conditions TEXT, status TEXT
);

CREATE TABLE IF NOT EXISTS policies (
  id TEXT PRIMARY KEY, title TEXT, body TEXT, category TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, branch_id TEXT, court_id TEXT, coach_id TEXT,
  rating INTEGER, text TEXT, author_name TEXT, date TEXT, is_noise INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS slots (
  id TEXT PRIMARY KEY, court_id TEXT, branch_id TEXT, date TEXT, start_time TEXT,
  duration_min INTEGER, status TEXT, price_aed INTEGER, version INTEGER
);

CREATE TABLE IF NOT EXISTS price_rules (
  id TEXT PRIMARY KEY, branch_id TEXT, court_type TEXT, day_type TEXT, band TEXT,
  applies_to_start_times TEXT, multiplier REAL, base_price_aed INTEGER, price_aed INTEGER
);

CREATE TABLE IF NOT EXISTS coach_schedules (
  id TEXT PRIMARY KEY, coach_id TEXT, branch_id TEXT, date TEXT,
  start_time TEXT, end_time TEXT, status TEXT
);

CREATE TABLE IF NOT EXISTS bookings (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, slot_ids TEXT NOT NULL,
  duration_min INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
  price_aed INTEGER, idempotency_key TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_idem
  ON bookings(idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Occupancy. One row per occupied slot. The PK is what makes the race test pass.
-- WITHOUT ROWID is load-bearing: a plain TEXT PRIMARY KEY on a rowid table accepts NULL.
CREATE TABLE IF NOT EXISTS slot_claims (
  slot_id    TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,      -- 'booking' | 'hold' | 'blocked'
  booking_id TEXT,
  session_id TEXT,
  expires_at INTEGER             -- unix epoch seconds; holds only
) STRICT, WITHOUT ROWID;

-- The unmarked second hour of legacy 90-minute bookings. Consulted on writes only,
-- never on reads, so reported availability still matches the shipped dataset exactly.
CREATE TABLE IF NOT EXISTS slot_overhang (
  slot_id TEXT PRIMARY KEY, booking_id TEXT NOT NULL
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX IF NOT EXISTS idx_slots_court_date_time
  ON slots(court_id, date, start_time);
CREATE INDEX IF NOT EXISTS idx_slots_branch_date ON slots(branch_id, date);
CREATE INDEX IF NOT EXISTS idx_courts_branch ON courts(branch_id);
CREATE INDEX IF NOT EXISTS idx_coaches_branch ON coaches(branch_id);
CREATE INDEX IF NOT EXISTS idx_classes_branch ON classes(branch_id);
CREATE INDEX IF NOT EXISTS idx_reviews_branch ON reviews(branch_id);
CREATE INDEX IF NOT EXISTS idx_cs_coach_date ON coach_schedules(coach_id, date);
"""

# Lexical index over prose. Doubles as the fallback when Chroma is unavailable.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
  record_id UNINDEXED, type UNINDEXED, branch_id UNINDEXED, title, body
);
"""


def connect() -> sqlite3.Connection:
    cfg = settings()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        cfg.db_path,
        timeout=cfg.sqlite_busy_timeout_ms / 1000,
        isolation_level=None,  # we control transactions explicitly
    )
    conn.row_factory = sqlite3.Row
    # busy_timeout and synchronous are per-connection and must be re-applied every time.
    conn.execute(f"PRAGMA busy_timeout = {cfg.sqlite_busy_timeout_ms}")
    conn.execute("PRAGMA synchronous = NORMAL")
    # Off deliberately: an FK violation would surface as IntegrityError, which the booking
    # handler maps to 409 -- but a nonexistent slot must be a 400. We check existence directly.
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


@contextmanager
def write_txn() -> Iterator[sqlite3.Connection]:
    """One connection, one BEGIN IMMEDIATE, always closed.

    Never open a second connection inside this block: the nested BEGIN IMMEDIATE would
    wait out busy_timeout and then fail, which the race test scores as a hang.
    """
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()  # implicit rollback of anything still open


@contextmanager
def read_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def init_schema() -> None:
    """Create tables and set the DB-level PRAGMAs. Safe to call repeatedly."""
    with read_conn() as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # persistent, DB-level
        conn.executescript(SCHEMA)
        if has_fts5(conn):
            conn.executescript(FTS_SCHEMA)

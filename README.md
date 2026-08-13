# Baseline Padel — booking and discovery assistant

A grounded conversational assistant over eight fictional padel branches in the UAE.
Answers questions in English and Arabic, books courts through conversation, and stays
correct under concurrent load.

---

## Run it

Requires **Python 3.12**. Node is **not** needed — the UI is committed pre-built.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && uvicorn app.main:app --port 3000
```

Then open **http://localhost:3000**.

> **The port is 3000, not 8000.** `tests/race.sh` defaults to `BASE_URL=http://localhost:3000`,
> so serving there means the race test runs with no environment variables at all.

The first start builds `data/padel.db` from the shipped JSON in under a second. Model
calls need a key: copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. Without
one the app still starts, and lexical search, availability, pricing and booking all keep
working — only the conversational layer needs a model.

To build the semantic index as well (about $0.012 of embeddings, one off):

```bash
python -m app.ingest --reset
```

---

## Run the eval

```bash
python -m app.eval --input eval/queries.json --output eval/results/latest.json
```

The harness **starts what it needs itself** — no server has to be running. `--input`
takes any path. Output is exactly the contract shape:
`[{query_id, retrieved_ids, answer, refused, latency_ms, cost_usd}]`.

Score a run against the expectations in `eval/gold.json`:

```bash
python -m app.score --results eval/results/latest.json --gold eval/gold.json
```

## Run the race test

```bash
python -m app.ingest --reset          # ← restores the slot; run this before each race
SLOT_ID=slot_alquoz_pc01_20260810_1800 bash tests/race.sh
```

`--reset` matters: the test books the slot, so a second run against an unreset database
correctly returns twenty conflicts and reports FAIL. Reset takes under a second and is
safe while the server is running.

```bash
python -m pytest tests/ -q            # 39 tests: concurrency, adjacency, degradation, agent
```

---

## Results

47 queries, 14 Arabic or mixed, 5 multi-turn. Measured on the shipped configuration.

| Metric | Result | Target |
| --- | --- | --- |
| Retrieval recall | 0.97–1.00 | — |
| Precision@1 / MRR | 0.75 / 0.80 | — |
| Refusal accuracy | **1.00** — no false and no missed refusals | scored both directions |
| PII leaks | **0** | 0 |
| Mean cost per query | **$0.0021** | ≤ $0.02 |
| p95 full response | **6.6 s** | ≤ 8 s |
| Time to first token | **2.8 s** | ≤ 2 s ✗ |
| Race test | 1 confirmation, 19 rejections, 5/5 runs | exactly 1 |

**Time to first token misses.** A turn is three sequential model round trips — plan,
decide which tool, then generate — and the first two produce no visible tokens. Profiling
cut the planner from 3.4 s to 1.0 s by moving it to a smaller model, which brought total
latency inside target, but the structure still costs about 2.8 s before the first word.
Removing it means either dropping the planner (which costs refusal accuracy, since scope
detection lives there) or running it concurrently with the first tool decision. The
second is the right fix and is not done.

Run-to-run variation of a few points is normal: temperature is 0, but which tools the
model chooses still varies.

---

## How it works

```
     React UI (ui/dist, committed)
              │ SSE
     ┌────────▼─────────┐
     │ FastAPI :3000    │   /api/v1/chat · /bookings · /holds · /slots
     └────────┬─────────┘
              │            same Python function, never an HTTP self-call
     ┌────────▼─────────────────┐
     │ LangGraph               │   plan → answer ⇄ tools
     └───┬──────────────────┬───┘
         │                  │
    ┌────▼─────┐      ┌─────▼──────┐
    │ SQLite   │      │  Chroma    │
    │ facts    │      │  prose     │
    │ + FTS5   │      └────────────┘
    │ + claims │
    └──────────┘
         └── LangSmith: every node, with latency, tokens and cost
```

**Retrieval splits by question shape.** Prose questions ("somewhere relaxed for beginners
with kids") go through Chroma and FTS5 fused with Reciprocal Rank Fusion. Facts
("is PC-07 free tomorrow at 7", "cheapest branch in the evening", "how many coaches in
Ajman") are SQL. A vector store cannot count, compare prices or read a calendar, and
embedding those questions would be slower, dearer and wrong.

**Booking correctness is a database constraint, not application logic.** A single
`slot_claims` table has `slot_id` as its primary key, so twenty concurrent writers all
try the same INSERT and SQLite lets exactly one through. Nothing depends on a Python
lock, so the guarantee survives multiple workers and processes. `WITHOUT ROWID` is
load-bearing: a plain `TEXT PRIMARY KEY` on a rowid table accepts NULL.

**Models are addressed by role, not vendor.** `LLM_PLANNER`, `LLM_RERANKER`,
`LLM_ANSWERER`, `LLM_FALLBACK` each take a `provider:model` string, so swapping a
component's model — or its provider — is one line of `.env`.

**Everything tunable lives in `app/config.py`**, environment-driven: hold TTL, top-k,
the RRF constant, busy timeout, allowed durations, per-model prices, and the reference
date.

---

## Decisions worth knowing

**The reference date is the dataset's, not the clock's.** Availability runs
2026-08-10 to 2026-08-24 against a reference date of 2026-08-09. "Tomorrow" resolves
from `dataset_meta.json`, so the eval set behaves identically whatever day you run it.

**The slot grid has a hole.** Courts run 06:00–10:00 and 15:00–23:00 — there are no
11:00–14:00 slots anywhere, though `opening_hours` claims 06:00–00:00. Adjacency for
multi-slot bookings is therefore a real `+60min` step on the same court and date, never
the next row in sort order. A 90-minute booking starting at 10:00 does not fit and
returns 400. The UI draws that gap rather than smoothing it away.

**`slots.price_aed` is the only trustworthy price.** `courts.price_per_hour_aed` has 5
nulls and 2 sentinel `99999` values; `price_rules.price_aed` disagrees with its own
`base × multiplier` in 86 of 120 rows and omits Al Ain indoor entirely.

**The 90-minute overhang.** 565 seeded bookings declare `duration_min: 90` while
referencing a single 60-minute slot; the second hour is unmarked, and 63 of those
neighbours are already blocked. We report availability exactly as the dataset states it —
claiming those hours would silently flip 502 slots that the data calls available — and
enforce the overhang on the write path, so no new booking can take a physically occupied
hour. A new 90-minute booking claims two contiguous slots and is priced at 1.5×, not 2×.

**Refusals are judged on behaviour, not intent.** "We hold no information about that" is
a refusal; "there is nothing free at that time" and "no, billing is not by instalment"
are answers. Markers are honoured only in the opening sentence, and questions of scope
are decided by the planner rather than string matching.

**Staff contact details never leave the retrieval layer.** `internal_phone` and
`internal_email` exist for all 45 coaches and are stripped from every record and excluded
from the search index. Branch phone numbers are public and are shared.

**Dirty data handled at ingest.** Reviews contain injected HTML and session-timeout
boilerplate plus duplicate bodies; those rows are kept (their ids may be in the graders'
key) but flagged out of the search index. Two dates are unparseable — `2026-13-07` and
`2026-02-30` — and are stored as NULL rather than crashing the load.

---

## The six challenge areas

All six were attempted.

| # | Area | Where it is |
| --- | --- | --- |
| 1 | **Slot holds** | `kind='hold'` with an integer-epoch TTL in the same claims table. Expiry is a scoped delete of only the requested slots inside the write transaction; a global sweep runs at startup. Confirming converts the hold in one transaction, and a session booking a slot it already holds converts rather than conflicting with itself. The UI drains a countdown bar on the held chip. |
| 2 | **Cross-turn references** | The session keeps the last ranked result list and the planner resolves "the second one", "the first one" or "same time at Yas" into concrete ids before retrieval. Five multi-turn eval cases (q39–q43, two of them Arabic) plus unit tests. Records carried from an earlier turn are counted as retrieved, since they are what grounds the answer. |
| 3 | **Graceful degradation** | Chroma down → FTS5 lexical, flagged. Reranker or planner down → the turn still answers. Provider down → the configured fallback vendor. Booking and all structured lookups are pure SQL and need no model. Eight tests in `tests/test_degradation.py`, each removing a real dependency. |
| 4 | **Cost reduction** | $0.0021 per query, roughly 10× under target. Profiling moved the planner to the smallest model, prose in tool payloads is capped (six policy bodies had pushed one call past 9,000 input tokens), and reviews are retrieved as a capped second pass. |
| 5 | **Multi-constraint booking** | `find_group_slots` handles party size → courts, simultaneity, adjacency and coach cover. Four eval cases (q44–q47) and eight tests covering the conversion, the adjacency run, simultaneity, and that no claimed or overhang slot is ever offered. Both soft constraints are surfaced as caveats rather than asserted, because the data records no court positions and does not link coaches to bookings. |
| 6 | **Reranking** | Built, measured, and **it does not improve results here** — precision@1 fell from 0.789 to 0.737 and p95 latency rose past target. It ships disabled with the evidence and the analysis in [`eval/RERANKING.md`](eval/RERANKING.md). |

---

## Observability

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. Every node is traced with
step, latency, input and output tokens, and cost. A captured per-node breakdown of a real
request is in [`docs/TRACING.md`](docs/TRACING.md); the dashboard is at
<https://smith.langchain.com> under project `baseline-padel`.

The same numbers are computed locally per request and returned on the `done` SSE event
and in the eval output, so `cost_usd` can be checked against the dashboard rather than
taken on trust. The UI prints them under each reply and turns time-to-first-token amber
when it exceeds 2 s.

![Chat with the hour rail](docs/ui-chat.png)

Above: the hour rail is the availability grid as it actually is — 06–10, a real break,
then 15–23. Teal hours are free and clickable, the amber chip is a live hold counting
down, and the trace strip carries the per-response numbers.

![Arabic reply](docs/ui-arabic.png)

---

## Layout

```
app/
  main.py            lifespan, router-before-static, 422→400
  config.py          all configuration, one place, env-driven
  db.py              connection, PRAGMAs, write_txn(), schema
  ingest.py          JSON → SQLite + Chroma        (python -m app.ingest)
  llm.py             role→model resolution, cost and latency ledger, fallback
  eval.py            harness                       (python -m app.eval)
  score.py           grading                       (python -m app.score)
  api/               bookings, slots, chat (SSE)
  services/          booking, retrieval, vectorstore
  agent/             graph, tools
eval/                queries.json, gold.json, RERANKING.md, results/
tests/               race.sh (provided), test_booking.py, test_degradation.py
ui/                  Vite React source; ui/dist is committed
```

## The brief

The original assessment brief is preserved at [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md),
and the plan this was built from is in [`plan.md`](plan.md).

## Known gaps

- Time to first token is 2.8 s against a 2 s target; see above for the cause and the fix.
- Reranking is implemented but disabled, because measurement says it hurts.
- A cross-encoder reranker was not tried: it needs `torch` and several hundred MB, against
  a clean-clone install requirement.
- Group booking places its courts in one transaction but does not hold them as a unit
  before confirmation.

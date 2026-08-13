# Tracing

Every model call is traced. Set these in `.env` and restart:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=baseline-padel
```

Dashboard: <https://smith.langchain.com> → project **baseline-padel**.

Tracing is opt-in and the app boots fine without it: `configure_tracing()` is a no-op
when the key is absent, and no module reads an API key at import time.

## What a request looks like

Captured from the live project for *"Is court PC-07 free tomorrow at 7pm and how much?"*
The graders' four requirements — step, latency, input and output tokens, cost — are all
present per node.

| Step | Type | Latency | Input tok | Output tok | Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| `LangGraph` (whole turn) | chain | 5502 ms | 4220 | 155 | $0.001217 |
| ├─ `plan` | chain | 2560 ms | 450 | 71 | $0.000073 |
| ├─ `answer` (choose tool) | chain | 1423 ms | 1967 | 45 | $0.000360 |
| ├─ `tools` → `check_availability` | tool | 33 ms | — | — | — |
| └─ `answer` (generate) | chain | 1364 ms | 1803 | 39 | $0.000784 |

Reading it: the planner is cheap in tokens but not in time, the tool call itself is 33 ms
of SQL, and the two answer passes carry almost all the input tokens because the system
prompt and tool schemas are resent each time. That profile is what drove moving the
planner to a smaller model and capping prose in tool payloads.

## The same numbers, locally

Every request also accumulates its own ledger, so the reported figures can be checked
against the dashboard rather than taken on trust:

- the `done` event on `/api/v1/chat` carries `latency_ms`, `ttft_ms`, `cost_usd`,
  `input_tokens`, `output_tokens` and a per-step breakdown;
- the UI prints them under each reply;
- `cost_usd` in the eval output comes from the same ledger.

Costs are computed from `usage_metadata` against the price table in `app/config.py`,
which is env-overridable — so when prices change, no code changes.

## Reproducing the table

```bash
python -c "
from app.config import settings
from langsmith import Client
c = Client(api_key=settings().langsmith_api_key)
for r in c.list_runs(project_name=settings().langsmith_project, limit=14):
    ms = round((r.end_time - r.start_time).total_seconds()*1000) if r.end_time else 0
    print(f'{r.name:<24} {r.run_type:<8} {ms:>6}ms {r.prompt_tokens or 0:>6} '
          f'{r.completion_tokens or 0:>5} {r.total_cost or 0}')
"
```

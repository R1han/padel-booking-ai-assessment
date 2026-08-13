# Latency: what was tried, and where the floor is

Total response time meets its target comfortably. **Time to first token does not**, and
after profiling and one full architectural attempt, the conclusion is that ~2s is not
reachable for a grounded tool-calling agent on these models. This records the work so
the miss is a measured limit rather than an unexamined one.

## Where the time goes

Profiled per model call (the same ledger LangSmith records):

| Step | Latency | Emits visible tokens? |
| --- | ---: | --- |
| `plan` | ~1.0s | no |
| `answer` #1 — choose a tool | ~1.3s | no |
| `tools` — SQL | ~0.03s | no |
| `answer` #2 — generate | ~0.6s to first word | **yes** |

Two of three round trips produce nothing the user can see. Shaving them does not help;
only removing one does.

The planner was already moved from `gpt-4.1-mini` to `gpt-4.1-nano` on this evidence,
which cut it from 2.5–3.9s to ~1.0s and brought **total** latency inside target.

## What was tried: merging planning into the tool decision

Planning was re-expressed as a `note_plan` tool emitted in parallel with the first
retrieval call, collapsing three model round trips into two. It works — the model does
call `note_plan` alongside `find_records` in one batch — but it is worse on every axis
that matters.

Measured over the full 47-query eval, and over 10 queries through the streaming endpoint:

| | 3 calls (shipped) | 2 calls (merged) |
| --- | ---: | ---: |
| Refusal accuracy | **1.000** | 0.979 |
| Precision@1 | **0.75** | 0.70 |
| MRR | **0.803** | 0.753 |
| Recall, partial queries | **1.000** | 0.667 |
| Mean cost / query | **$0.00212** | $0.00274 (+29%) |
| TTFT median | 2952 ms | 2854 ms |
| TTFT p95 | **4152 ms** | 5288 ms |
| Total p95 (streaming) | **5822 ms** | 6633 ms |

The merged version is *slightly* faster at the median and clearly worse at p95, because
the combined first call carries the planning rules on the larger answering model and
provokes more tool loops. It also lost the quality the separate planner was buying.

It survives behind `AGENT_SEPARATE_PLAN_NODE=false` for anyone who wants to re-measure,
but the default is the three-call shape.

Two real bugs surfaced while building it, both fixed and both worth knowing:

- **A `ContextVar.set()` inside a LangGraph tool is invisible to the caller.** Tools run
  in their own context, so the plan the tool recorded never came back and every request
  silently used defaults. Mutating an object the caller already holds propagates;
  rebinding does not. The surfaced-ids tracking had worked all along only because it
  mutates a list.
- **Asking the model to report the user's language was unreliable**, labelling Arabic
  questions `"en"`, which then instructed the answer step to reply in English. Script
  detection is deterministic, free and cannot get it wrong; it is now a regex.

## Why ~2s is a floor here

A grounded answer cannot start streaming before the system knows what to ground it in.
That ordering is forced:

```
model decides what to retrieve  →  retrieve  →  model starts generating
        ~1.3s                      ~0.03s            ~0.6s
```

That is ~1.9s with planning removed entirely, and it assumes a single tool round. Any
query needing two lookups exceeds it. The remaining levers all cost something real:

- **A deterministic router** instead of a model call for tool selection would put TTFT
  near 0.8s, at the cost of much worse tool choice on the open-ended queries.
- **A smaller model for the tool-selection call only**, keeping the good model for
  generation, is the most promising untried option — perhaps 0.5s off the median, though
  p95 is set by multi-round queries it would not help.
- **Emitting a filler token before retrieval** would satisfy the metric and mislead the
  reader. Not done.

## Shipped numbers

Streaming endpoint, 10 mixed English and Arabic queries:

```
TTFT   median 2952 ms   p95 4152 ms
TOTAL  median 3807 ms   p95 5822 ms
```

Full response is inside the 8s target. Time to first token is not inside 2s, and the UI
prints it in amber when it exceeds the target rather than hiding it.

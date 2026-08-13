# AI Engineer | Technical Assessment

## Overview

Build a booking and discovery assistant for a fictional padel club group operating eight branches across the UAE.

Users should be able to ask about branches, courts, coaches, classes, packages and policies in natural language, and book a court through conversation.

**Deliverable:** a working system in a git repository

---

## AI tooling

You are encouraged to use it. Cursor, Claude Code, Copilot, or whatever you rely on in your normal work.

During our check-in conversations we will ask you to close AI tools and talk through your code directly.

---



## The data

You will receive a repository containing:

```
catalog/            branches, courts, coaches, classes, packages, reviews, policies
structured/         slots, bookings, price rules, coach schedules
eval/               15 seed queries
dataset_meta.json   reference date and record counts
```

The data was assembled from mixed sources.

Availability data covers a rolling 15-day window.

The catalog is in English. User queries will be in English, Arabic, and mixed. This is a UAE product.

---



## Required scope



### 1. Ingest

Load the data into whatever stores you choose. Decide what belongs where.

### 2. Retrieval

Users ask things like *"somewhere relaxed for beginners with kids"*, *"is court PC-07 free tomorrow at 7"*, and *"ابغى حصة مع Coach Marwan"*. Your retrieval needs to serve all of these.

### 3. Conversational agent

Grounded in the data. The system must:

- Answer only from retrieved context
- State clearly when it does not know, or when a request falls outside scope
- Never invent a branch, coach, court, price, or availability



### 4. Booking

Users book through conversation. Booking must remain correct under concurrent load.

A race-test script ships with this repository (`EVAL_CONTRACT.md`, Part 2). We will run it against your system. It issues 20 simultaneous booking requests for the same slot.

> **Required result: exactly 1 confirmation and 19 clean rejections.**
> Two confirmations for the same slot is an automatic fail.

Bookings vary in duration. Not every booking occupies a single slot.

### 5. Evaluation

We provide 15 seed queries across three categories: answerable, partially answerable, and unanswerable.

**Extend the set to at least 30 queries, of which at least 8 are Arabic or mixed.** For each, record what a correct system should retrieve and how it should behave.

Your eval harness must be runnable by us with a single command, against a query file we supply. The interface contract is specified in `EVAL_CONTRACT.md` and should be followed exactly. **We will run our own query set against your system on the final day.**

### 6. Observability

Every model call must be traced. We should be able to open a dashboard and see, per request:

- Step or node
- Latency
- Input and output tokens
- Cost

How you achieve this is your decision. Be prepared to walk us through a trace.

### 7. Chat interface

A minimal streaming chat page. No authentication, no user accounts, and no persisted history beyond the active session.

The requirement is that responses stream and the booking flow works end to end.

---



## Pick three of six

Beyond the required scope, choose three areas and develop them properly. Tell us on day one which three you have chosen.


| #   | Challenge                | Description                                                                                           |
| --- | ------------------------ | ----------------------------------------------------------------------------------------------------- |
| 1   | Slot holds               | Reserve a slot when the assistant proposes it, and release it on a timeout if the user never confirms |
| 2   | Cross-turn references    | Handle "book the second one you mentioned" or "the one at Yas, same time"                             |
| 3   | Graceful degradation     | The system stays useful when a dependency becomes unavailable                                         |
| 4   | Cost reduction           | Bring measured cost per query meaningfully below the target without losing eval accuracy              |
| 5   | Multi-constraint booking | "Two courts side by side for eight of us on Friday evening, with a coach"                             |
| 6   | Reranking                | Add a reranking stage and demonstrate on your eval set that it improves results                       |


**You may also propose your own.** If you believe something matters more than anything on this list, pitch it to us on day one. A well-argued proposal counts the same as a listed option.

We would rather see three areas developed thoroughly than six covered superficially.

---



## Technical requirements



### Stack

Either:


|          | Node                            | Python                                  |
| -------- | ------------------------------- | --------------------------------------- |
| Version  | 24                              | 3.12                                    |
| Language | TypeScript, `strict: true`      |                                         |
| Packages | npm, commit `package-lock.json` | `requirements.txt` with pinned versions |
| Server   | your choice                     | FastAPI and uvicorn                     |


Vector store, database and cache are your choice. Local installations, Docker, or free tiers are all acceptable. Use Docker only if you need it; containerization is not a requirement.

### Model access

We provide OpenAI and Anthropic API keys for the week. These are revoked afterwards.

**Cost per query is graded, and usage is visible on our account.**

You may use other providers where you can justify the choice.

### Model abstraction

Model calls must go through an abstraction that allows providers to be swapped **through configuration rather than code changes**. We should be able to change which model a component uses in one place and have the system work.

### Configuration

- No credentials in the repository. We scan for these.
- Configuration held in one place, environment driven, with a committed `.env.example`
- No magic numbers embedded in business logic



### It must run

From a clean clone, using one documented command:

```
npm ci && npm run dev
```

```
pip install -r requirements.txt && uvicorn app.main:app
```

A system that does not start on our machine cannot be assessed. Please test this against a fresh clone before the final day, and document your port.

---



## Performance targets


| Metric              | Target                                       |
| ------------------- | -------------------------------------------- |
| Time to first token | 2s or less (p95)                             |
| Full response       | 8s or less                                   |
| Cost per query      | $0.02 or less, averaged across your eval set |


---



## Pass/fail gates

- [ ] Clean clone runs with one documented command
- [ ] No credentials committed to the repository
- [ ] Race test produces exactly one confirmation
- [ ] Tracing shows per-request latency, tokens and cost
- [ ] Eval harness runs on our query file and reproduces your reported numbers
- [ ] You can explain your own code without AI assistance

---



## Scoring


| Criterion                                    | Weight |
| -------------------------------------------- | ------ |
| Retrieval, ingest, and evaluation            | 25%    |
| Booking correctness under concurrency        | 20%    |
| Grounding and refusal behaviour              | 15%    |
| Architecture and code organisation           | 15%    |
| Check-in conversations and live modification | 15%    |
| Cost, latency, and observability             | 10%    |


Your three chosen challenges are graded within the criteria they touch.

---



## Check-ins

We will sit with you several times across the week, for roughly 30 minutes on each occasion. No slides or preparation are required.

On the final day we will ask you to make a small modification while we watch.

---



## Submission

- Git repository with meaningful commit history
- `README.md` covering setup, run instructions, and how to run your eval
- Your eval query set and results
- A link or screenshot of your tracing dashboard

No slide deck, demo video, or written report is required.

---



## Questions

Please ask. We sit nearby, and we would rather answer a question early than see time lost to a wrong assumption.
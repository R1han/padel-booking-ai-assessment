# Eval Contract and Race Test

Two interfaces your system must implement so that we can run our own tests against it.

Both are small by design. If either does not work on the final day, the components that depend on them cannot be assessed.

---

## Part 1: Eval harness contract

On the final day we run our own query set against your system, in addition to the set you build.

### Command

A single command taking two arguments:

```
npm run eval -- --input <path/to/queries.json> --output <path/to/results.json>
```

```
python -m app.eval --input <path/to/queries.json> --output <path/to/results.json>
```

Document the exact command in your README. It must accept an arbitrary input path. Do not hardcode a filename.

### Input format

We supply this file. It uses the same shape as the seed set you received.

```json
[
  {
    "query_id": "q01",
    "query": "Which branches have indoor courts?"
  },
  {
    "query_id": "q02",
    "query": "كم كوتش عندكم في فرع عجمان؟"
  }
]
```

Assume 30 to 40 queries. Assume some are in Arabic. Assume some have no answer in the data.

### Output format

```json
[
  {
    "query_id": "q01",
    "retrieved_ids": ["br_alquoz", "br_jvc", "br_yas"],
    "answer": "<your system's user-facing reply, as a single string>",
    "refused": false,
    "latency_ms": 1840,
    "cost_usd": 0.0113
  }
]
```

The values above illustrate the shape only. They are not a worked example of any
particular query.


| Field           | Type     | Notes                                                                                                        |
| --------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `query_id`      | string   | Echo the input value                                                                                         |
| `retrieved_ids` | string[] | Record IDs from the dataset, ranked, best first. Whatever your retrieval surfaced before generation          |
| `answer`        | string   | The final user-facing response                                                                               |
| `refused`       | boolean  | `true` where the system declined to answer, whether for scope, insufficient information, or any other reason |
| `latency_ms`    | number   | End to end, per query                                                                                        |
| `cost_usd`      | number   | Total for the query, including all model calls                                                               |


`retrieved_ids` **must use the dataset's own record IDs.** If you generate your own identifiers during ingest, map them back before writing results. We compare against a key built on the original IDs, so remapped or generated IDs will not match.

`refused` is evaluated in both directions.

### Requirements

- The harness runs against a running system, or starts what it needs itself. Document which.
- It handles a query it cannot answer without crashing.
- The numbers it produces match the numbers you report to us.
- Output order does not matter. We match on `query_id`.

---



## Part 2: Booking endpoint and race test



### Required endpoint

Your system must expose a booking endpoint that can be called directly. Your conversational agent should route through the same underlying logic, with the chat layer sitting in front of it rather than beside it.

```
POST /api/v1/bookings
Content-Type: application/json

{
  "slot_ids": ["slot_alquoz_pc01_20260810_1800"],
  "user_id": "usr_0042",
  "duration_min": 60
}
```

**Success,** `201`

```json
{
  "booking_id": "bkg_01H...",
  "status": "confirmed",
  "slot_ids": ["slot_alquoz_pc01_20260810_1800"]
}
```

**Conflict,** `409`

```json
{
  "error": "slot_unavailable",
  "message": "That slot was taken.",
  "slot_ids": ["slot_alquoz_pc01_20260810_1800"]
}
```

**Invalid,** `400` for a slot that does not exist, a duration that does not fit, or a malformed request.

### The test

```
BASE_URL=http://localhost:3000 \
SLOT_ID=slot_alquoz_pc01_20260810_1800 \
./tests/race.sh
```

The script issues 20 simultaneous booking requests for the same slot and counts the responses.

### Pass condition


| Outcome                                     | Result         |
| ------------------------------------------- | -------------- |
| Exactly one `201`, with the remainder `409` | Pass           |
| Two or more confirmations                   | Automatic fail |
| Any `5xx` response                          | Fail           |
| The request hangs or deadlocks              | Fail           |


Rejections must return `409`. A `5xx` response is treated as a failure.

### Additional checks

We also run the following against your endpoint:


| Check                                                       | Expected                           |
| ----------------------------------------------------------- | ---------------------------------- |
| A 90 minute booking against an already booked adjacent slot | Rejected                           |
| A booking for a slot ID that is not in the dataset          | `400`, and nothing created         |
| Reading the slot back after the race test                   | State reflects exactly one booking |


---



## Not required

- Authentication on the booking endpoint. `user_id` in the body is sufficient.
- Idempotency keys, unless you want them.
- Any particular locking strategy. Optimistic, pessimistic, or database level are all acceptable.
- Any particular database.

The constraint is the outcome: one slot, one booking, under concurrent load.
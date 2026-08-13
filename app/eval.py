"""Eval harness.

    python -m app.eval --input <queries.json> --output <results.json>

Runs in-process and builds whatever it needs, so no server has to be running. The input
path is arbitrary; nothing is hardcoded.

Output is exactly the contract shape:
    [{query_id, retrieved_ids, answer, refused, latency_ms, cost_usd}]

retrieved_ids are the dataset's own record ids, in the order retrieval surfaced them.
A query that fails is still reported -- as a refusal with the error in `answer` -- because
the harness must not crash on a question it cannot answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("padel.eval")

DEFAULT_CONCURRENCY = 4


def load_queries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # tolerate {"queries": [...]}
        data = data.get("queries", [])
    queries = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            item = {"query_id": f"q{i + 1:02d}", "query": item}
        queries.append({
            "query_id": str(item.get("query_id") or f"q{i + 1:02d}"),
            "query": item.get("query") or item.get("text") or "",
        })
    return queries


async def run_one(item: dict, semaphore: asyncio.Semaphore) -> dict:
    from app.agent.graph import run_query

    async with semaphore:
        started = time.perf_counter()
        try:
            result = await run_query(item["query"], session_id=f"eval-{item['query_id']}")
            return {
                "query_id": item["query_id"],
                "retrieved_ids": result["retrieved_ids"],
                "answer": result["answer"],
                "refused": result["refused"],
                "latency_ms": result["latency_ms"],
                "cost_usd": result["cost_usd"],
            }
        except Exception as exc:  # noqa: BLE001 - one bad query must not stop the run
            log.exception("query %s failed", item["query_id"])
            return {
                "query_id": item["query_id"],
                "retrieved_ids": [],
                "answer": f"The system could not answer this request ({exc}).",
                "refused": True,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "cost_usd": 0.0,
            }


async def run_all(queries: list[dict], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [run_one(item, semaphore) for item in queries]
    results = []
    for done in asyncio.as_completed(tasks):
        result = await done
        results.append(result)
        print(f"  {result['query_id']:>5}  {result['latency_ms']:>6}ms  "
              f"${result['cost_usd']:.4f}  {'refused' if result['refused'] else 'answered'}",
              file=sys.stderr, flush=True)
    return results


def summarise(results: list[dict]) -> dict:
    if not results:
        return {}
    latencies = sorted(r["latency_ms"] for r in results)
    costs = [r["cost_usd"] for r in results]
    return {
        "queries": len(results),
        "refused": sum(1 for r in results if r["refused"]),
        "mean_cost_usd": round(sum(costs) / len(costs), 5),
        "total_cost_usd": round(sum(costs), 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies)),
        "p95_latency_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)],
        "max_latency_ms": latencies[-1],
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m app.eval", description="Run a query file through the assistant."
    )
    parser.add_argument("--input", required=True, type=Path, help="path to queries JSON")
    parser.add_argument("--output", required=True, type=Path, help="path to write results")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()

    from app import db, llm
    from app.ingest import ingest, is_ingested

    llm.configure_tracing()
    db.init_schema()
    if not is_ingested():
        print("database empty, ingesting first...", file=sys.stderr)
        ingest()

    queries = load_queries(args.input)
    print(f"running {len(queries)} queries from {args.input}", file=sys.stderr)

    started = time.perf_counter()
    results = asyncio.run(run_all(queries, args.concurrency))
    elapsed = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = summarise(results)
    print(f"\nwrote {len(results)} results to {args.output} in {elapsed:.1f}s", file=sys.stderr)
    for key, value in stats.items():
        print(f"  {key}: {value}", file=sys.stderr)


if __name__ == "__main__":
    main()

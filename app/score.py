"""Grade eval results against the gold file.

    python -m app.score --results <results.json> --gold eval/gold.json

Reports retrieval recall, refusal accuracy in both directions (the contract scores it
both ways), fact checks, and the cost and latency numbers against the stated targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_COST_USD = 0.02
TARGET_P95_TTFT_MS = 2000
TARGET_FULL_MS = 8000


def contains(answer: str, needle: str) -> bool:
    return needle.lower() in (answer or "").lower()


def grade(results: list[dict], gold: dict) -> dict:
    expected = gold["queries"]
    by_id = {r["query_id"]: r for r in results}

    rows, recalls = [], []
    precision_at_1: list[float] = []
    reciprocal_ranks: list[float] = []
    refusal_ok = refusal_total = 0
    false_refusals: list[str] = []
    missed_refusals: list[str] = []
    fact_failures: list[str] = []
    leak_failures: list[str] = []

    for query_id, spec in expected.items():
        result = by_id.get(query_id)
        if result is None:
            rows.append({"query_id": query_id, "status": "MISSING"})
            continue

        ranked = result.get("retrieved_ids") or []
        retrieved = set(ranked)
        wanted = set(spec.get("expected_ids") or [])
        recall = len(wanted & retrieved) / len(wanted) if wanted else None
        if recall is not None:
            recalls.append(recall)
            # Recall alone saturates once k is generous, so it cannot tell whether
            # reranking put the *best* record first. These two can.
            precision_at_1.append(1.0 if ranked and ranked[0] in wanted else 0.0)
            rank = next((i + 1 for i, rid in enumerate(ranked) if rid in wanted), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        refusal_total += 1
        want_refusal = bool(spec.get("should_refuse"))
        got_refusal = bool(result.get("refused"))
        if want_refusal == got_refusal:
            refusal_ok += 1
        elif got_refusal:
            false_refusals.append(query_id)
        else:
            missed_refusals.append(query_id)

        answer = result.get("answer", "")
        missing_facts = [f for f in spec.get("must_contain", []) if not contains(answer, f)]
        leaks = [f for f in spec.get("must_not_contain", []) if contains(answer, f)]
        if missing_facts:
            fact_failures.append(f"{query_id}: missing {missing_facts}")
        if leaks:
            leak_failures.append(f"{query_id}: leaked {leaks}")

        rows.append({
            "query_id": query_id,
            "category": spec.get("category"),
            "recall": None if recall is None else round(recall, 2),
            "refusal": "ok" if want_refusal == got_refusal else
                       ("false refusal" if got_refusal else "missed refusal"),
            "facts": "ok" if not missing_facts else f"missing {missing_facts}",
            "leak": "none" if not leaks else f"LEAKED {leaks}",
            "latency_ms": result.get("latency_ms"),
            "cost_usd": result.get("cost_usd"),
        })

    latencies = sorted(r["latency_ms"] for r in results if r.get("latency_ms"))
    costs = [r.get("cost_usd", 0) for r in results]
    by_category: dict[str, list[float]] = {}
    for query_id, spec in expected.items():
        if query_id in by_id and spec.get("expected_ids"):
            result = by_id[query_id]
            wanted = set(spec["expected_ids"])
            got = set(result.get("retrieved_ids") or [])
            by_category.setdefault(spec.get("category", "?"), []).append(
                len(wanted & got) / len(wanted)
            )

    return {
        "rows": rows,
        "summary": {
            "queries_scored": len(by_id),
            "retrieval_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
            "precision_at_1": round(sum(precision_at_1) / len(precision_at_1), 3)
                              if precision_at_1 else None,
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3)
                   if reciprocal_ranks else None,
            "recall_by_category": {
                k: round(sum(v) / len(v), 3) for k, v in sorted(by_category.items())
            },
            "refusal_accuracy": round(refusal_ok / refusal_total, 3) if refusal_total else None,
            "false_refusals": false_refusals,
            "missed_refusals": missed_refusals,
            "fact_failures": fact_failures,
            "pii_leaks": leak_failures,
            "mean_cost_usd": round(sum(costs) / len(costs), 5) if costs else None,
            "total_cost_usd": round(sum(costs), 4),
            "mean_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "p95_latency_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None,
        },
    }


def report(graded: dict) -> None:
    summary = graded["summary"]
    print(f"{'query':>6}  {'category':<12} {'recall':>6}  {'refusal':<15} {'facts':<28} "
          f"{'latency':>8} {'cost':>8}")
    print("-" * 96)
    for row in graded["rows"]:
        if row.get("status") == "MISSING":
            print(f"{row['query_id']:>6}  MISSING FROM RESULTS")
            continue
        recall = "-" if row["recall"] is None else f"{row['recall']:.2f}"
        leak = "" if row["leak"] == "none" else f"  {row['leak']}"
        print(f"{row['query_id']:>6}  {row['category'] or '?':<12} {recall:>6}  "
              f"{row['refusal']:<15} {row['facts'][:28]:<28} "
              f"{row['latency_ms']:>7}ms {row['cost_usd']:>8.4f}{leak}")

    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nTARGETS")
    cost = summary["mean_cost_usd"] or 0
    p95 = summary["p95_latency_ms"] or 0
    print(f"  mean cost / query   {cost:.5f}  target <= {TARGET_COST_USD}   "
          f"{'PASS' if cost <= TARGET_COST_USD else 'FAIL'}")
    print(f"  p95 full response   {p95}ms  target <= {TARGET_FULL_MS}ms  "
          f"{'PASS' if p95 <= TARGET_FULL_MS else 'FAIL'}")
    print(f"  PII leaks           {len(summary['pii_leaks'])}  target 0   "
          f"{'PASS' if not summary['pii_leaks'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.score")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.json"))
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    graded = grade(results, gold)

    if args.json:
        print(json.dumps(graded, ensure_ascii=False, indent=2))
    else:
        report(graded)


if __name__ == "__main__":
    main()

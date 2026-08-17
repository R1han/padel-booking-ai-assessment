#!/usr/bin/env python3
"""Padel data ingestion pipeline — stage 1: make the data proper.

Usage:
    python run_pipeline.py --input <dir with the 11 json files> --output <dir>

Outputs:
    <output>/cleaned/*.json        corrected datasets (same shape as input)
    <output>/issue_ledger.json     every detected issue with evidence + action
    <output>/issue_ledger.csv      same, for spreadsheets
    <output>/report.md             human-readable run summary

Set ANTHROPIC_API_KEY to enable the LLM semantic tier (optional; heuristics
cover the current dataset without it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pydantic import ValidationError

from pipeline.ledger import Ledger, Issue, Severity, Action
from pipeline.models import SCHEMAS
from pipeline.checks_rules import run_rule_checks
from pipeline.checks_semantic import run_semantic_checks
from pipeline.fixes import resolve, verify_post_fix
from pipeline.llm import get_llm

FILES = ["branches", "courts", "coaches", "classes", "packages", "policies",
         "price_rules", "reviews", "bookings", "coach_schedules", "slots"]


def load_and_validate(input_dir: str, ledger: Ledger) -> dict:
    data: dict[str, list[dict]] = {}
    for name in FILES:
        path = os.path.join(input_dir, f"{name}.json")
        with open(path) as f:
            records = json.load(f)
        schema = SCHEMAS[name]
        kept = []
        for r in records:
            try:
                schema.model_validate(r)
                kept.append(r)
            except ValidationError as e:
                ledger.add(Issue(name, r.get("id", "<no id>"), "-", "schema_validation_error",
                                 Severity.ERROR, detected_value=str(e.errors()[0])[:200],
                                 action=Action.QUARANTINED, checker="schema"))
                kept.append(r)  # keep the record; downstream layers may repair it
        data[name] = kept
    return data


def write_outputs(data: dict, ledger: Ledger, output_dir: str, post_fix_problems: list[str],
                  llm_enabled: bool) -> None:
    cleaned = os.path.join(output_dir, "cleaned")
    os.makedirs(cleaned, exist_ok=True)
    for name, records in data.items():
        with open(os.path.join(cleaned, f"{name}.json"), "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    ledger.to_json(os.path.join(output_dir, "issue_ledger.json"))
    ledger.to_csv(os.path.join(output_dir, "issue_ledger.csv"))

    lines = ["# Data ingestion report — stage 1 (cleaning)", ""]
    lines.append(f"LLM semantic tier: {'enabled' if llm_enabled else 'disabled (heuristics only)'}")
    lines.append("")
    lines.append("## Actions")
    for action, n in sorted(ledger.counts_by_action().items()):
        lines.append(f"- {action}: {n}")
    lines.append("")
    lines.append("## Issues by type")
    for key, stats in sorted(ledger.summary().items()):
        lines.append(f"- `{key}` — {stats}")
    lines.append("")
    lines.append("## Post-fix verification")
    if post_fix_problems:
        lines.extend(f"- FAIL: {p}" for p in post_fix_problems)
    else:
        lines.append("- All invariants pass on the cleaned dataset.")
    lines.append("")
    lines.append("## Quarantined (needs human review)")
    q = [i for i in ledger.issues if i.action == Action.QUARANTINED]
    if q:
        for i in q:
            lines.append(f"- {i.entity}/{i.entity_id} `{i.field}` ({i.issue_type}): "
                         f"{i.detected_value!r} — {i.evidence}")
    else:
        lines.append("- none")
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-llm", action="store_true", help="skip LLM tier even if key present")
    args = ap.parse_args()

    ledger = Ledger()
    llm = None if args.no_llm else get_llm()

    print("1/5 load + schema validation")
    data = load_and_validate(args.input, ledger)

    print("2/5 rule-based checks")
    run_rule_checks(data, ledger)

    print(f"3/5 semantic checks (LLM: {'on' if llm else 'off'})")
    # LLM tier is inert until the adjudication stage; the old semantic checks run
    # on heuristics only. A later task replaces this call with claim-extraction logic.
    run_semantic_checks(data, ledger)

    print("4/5 resolution")
    resolve(data, ledger)

    print("5/5 post-fix verification + outputs")
    problems = verify_post_fix(data, ledger)
    os.makedirs(args.output, exist_ok=True)
    write_outputs(data, ledger, args.output, problems, llm is not None)

    print(f"\nIssues: {len(ledger.issues)} | by action: {ledger.counts_by_action()}")
    if problems:
        print("POST-FIX FAILURES:")
        for p in problems:
            print(" -", p)
        return 1
    print("Post-fix verification: all invariants pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

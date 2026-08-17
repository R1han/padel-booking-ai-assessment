"""Issue ledger: every anomaly the pipeline sees is recorded here — nothing is fixed silently."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Severity(str, Enum):
    ERROR = "error"        # would corrupt downstream answers if inserted as-is
    WARNING = "warning"    # suspicious, may be acceptable
    INFO = "info"          # validated / explained, no action needed


class Action(str, Enum):
    AUTO_FIXED = "auto_fixed"          # pipeline corrected it, evidence + confidence recorded
    QUARANTINED = "quarantined"        # needs human review; record still ingested with flag (or held)
    VALIDATED_OK = "validated_ok"      # looked suspicious but was verified acceptable
    UNRESOLVED = "unresolved"          # detected, no safe fix available


@dataclass
class Issue:
    entity: str                 # e.g. "courts"
    entity_id: str              # e.g. "crt_alain_sc04"
    field: str                  # e.g. "type"
    issue_type: str             # e.g. "semantic_type_description_mismatch"
    severity: Severity
    detected_value: Any
    corrected_value: Any = None
    action: Action = Action.UNRESOLVED
    evidence: str = ""          # human-readable justification
    confidence: float | None = None  # 0..1 for fixes
    checker: str = ""           # which check produced this
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Ledger:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, issue: Issue) -> Issue:
        self.issues.append(issue)
        return issue

    # -- reporting -------------------------------------------------------
    def summary(self) -> dict:
        out: dict[str, dict[str, int]] = {}
        for i in self.issues:
            key = f"{i.entity}.{i.field}::{i.issue_type}"
            bucket = out.setdefault(key, {"count": 0})
            bucket["count"] += 1
            bucket[i.action.value] = bucket.get(i.action.value, 0) + 1
        return out

    def counts_by_action(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for i in self.issues:
            c[i.action.value] = c.get(i.action.value, 0) + 1
        return c

    # -- persistence -----------------------------------------------------
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([self._row(i) for i in self.issues], f, indent=2, default=str)

    def to_csv(self, path: str) -> None:
        rows = [self._row(i) for i in self.issues]
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                r = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
                w.writerow(r)

    @staticmethod
    def _row(i: Issue) -> dict:
        d = asdict(i)
        d["severity"] = i.severity.value
        d["action"] = i.action.value
        return d

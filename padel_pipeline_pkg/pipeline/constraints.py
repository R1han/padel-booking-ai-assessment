"""Cross-entity constraints. These veto claims; they never overrule silently.

A single description gives you a claim. A second, independent description
gives you adjudication. Branch descriptions state each branch's indoor/outdoor
court split in prose, which is an independent check on every court's `type` —
and on the shipped data it settles all eight of the disputed courts.
"""
from __future__ import annotations

from collections import Counter

from pydantic import BaseModel


def court_type_verdicts(
    data: dict,
    proposed: dict[str, str],
    branch_claims: dict[str, BaseModel],
) -> dict[str, str | None]:
    """Judge proposed court-type changes against each branch's stated split.

    The constraint is evaluated over the CANDIDATE assignment — shipped types
    with the proposals applied — so it tests the outcome, not the input.

    Returns {court_id: rejection reason or None}. Courts whose branch does not
    state a split are accepted (None): absence of evidence is not evidence.
    """
    courts_by_branch: dict[str, list[dict]] = {}
    for c in data.get("courts", []):
        courts_by_branch.setdefault(c["branch_id"], []).append(c)

    verdicts: dict[str, str | None] = {}
    for branch_id, courts in courts_by_branch.items():
        touched = [c for c in courts if c["id"] in proposed]
        if not touched:
            continue

        claim = branch_claims.get(branch_id)
        stated_in = claim.indoor_courts if claim else None
        stated_out = claim.outdoor_courts if claim else None
        if not (stated_in and stated_in.stated and stated_out and stated_out.stated):
            for c in touched:
                verdicts[c["id"]] = None
            continue

        candidate = Counter(proposed.get(c["id"], c["type"]) for c in courts)
        want_in, want_out = stated_in.value, stated_out.value
        if candidate["indoor"] == want_in and candidate["outdoor"] == want_out:
            for c in touched:
                verdicts[c["id"]] = None
        else:
            reason = (
                f"branch {branch_id} description states {want_in} indoor / "
                f"{want_out} outdoor ({stated_in.evidence!r}); this change would "
                f"give {candidate['indoor']} indoor / {candidate['outdoor']} outdoor"
            )
            for c in touched:
                verdicts[c["id"]] = reason
    return verdicts

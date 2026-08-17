"""Is this text evidence-bearing prose, or injected junk?

Three court descriptions and three class descriptions in the shipped data are
scraper noise — JavaScript errors, nav breadcrumbs, raw HTML — that happen to
contain keyword-stuffed terms like "indoor court". Treating those as evidence
flips genuinely outdoor courts to indoor. Nothing is extracted from text that
fails this gate.
"""
from __future__ import annotations

import re

# ponytail: duplicated in spirit with NOISE_PATTERNS in app/ingest.py, which
# screens noisy reviews out of the search index. Different job, different
# package, three lines — unify only if a third consumer appears.
NOISE_PATTERNS = re.compile(
    r"<[a-z]+[^>]*>"                    # raw HTML tags
    r"|session (has )?expired"
    r"|log in again"
    r"|loading avail"
    r"|javascript is required"
    r"|skip to main content"
    r"|error code"
    r"|^\s*home\s*>",                    # nav breadcrumb
    re.IGNORECASE,
)

MIN_PROSE_CHARS = 120


def is_prose(text: str | None) -> tuple[bool, str]:
    """Return (usable, reason). reason is 'ok' when usable."""
    if not text or not text.strip():
        return False, "empty text"
    if len(text.strip()) < MIN_PROSE_CHARS:
        return False, f"under {MIN_PROSE_CHARS} chars"
    m = NOISE_PATTERNS.search(text)
    if m:
        return False, f"injected noise: {m.group(0)[:40]!r}"
    return True, "ok"

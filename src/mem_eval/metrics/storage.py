"""Metric 7 — storage growth over time (PRD §7.7).

stats().bytes and item_count sampled as a function of #sessions ingested:
footprint at N sessions and a growth slope (bytes per session). Operational
metric: excluded from the determinism hash.
"""

from __future__ import annotations


def storage_growth(samples: list[tuple[int, int, int]]) -> dict:
    """samples: list of (sessions_ingested, bytes, item_count) checkpoints."""
    bytes_at_n = {str(n): b for n, b, _c in samples}
    items_at_n = {str(n): c for n, _b, c in samples}
    slope = 0.0
    if len(samples) >= 2:
        (n0, b0, _), (n1, b1, _) = samples[0], samples[-1]
        if n1 != n0:
            slope = (b1 - b0) / (n1 - n0)
    return {
        "bytes_at_n": bytes_at_n,
        "items_at_n": items_at_n,
        "growth_slope": slope,
    }


__all__ = ["storage_growth"]

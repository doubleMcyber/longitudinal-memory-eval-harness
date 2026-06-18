"""Metric 5 — query latency (PRD §7.5): wall-clock ms per query as p50/p95/p99.

Operational metric: reported and compared but excluded from the determinism hash
(PRD §7 'pass/fail vs reported', §10.3).
"""

from __future__ import annotations

from mem_eval.metrics.records import EvalRecord


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # nearest-rank percentile
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def latency_percentiles(records: list[EvalRecord]) -> dict[str, float]:
    vals = sorted(r.latency_ms for r in records)
    return {
        "p50": _percentile(vals, 50),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
    }


__all__ = ["latency_percentiles"]

"""Metric 4 — staleness (PRD §7.4).

(# returned items that are superseded as_of) / (# returned items), averaged over
staleness-probe queries. Lower is better. This is the metric NaiveRAG must fail
(staleness > 0) on the contradiction suite — it has no mechanism to prefer fresh
facts. Probes are the queries that carry superseded gold (the contradiction set).
"""

from __future__ import annotations

from mem_eval.metrics.records import EvalRecord


def staleness(records: list[EvalRecord]) -> float:
    probes = [r for r in records if r.gold_superseded]
    if not probes:
        return 0.0
    vals = []
    for r in probes:
        if r.retrieved_count == 0:
            # Convention: returning nothing scores 0 staleness (no stale item was
            # surfaced). This means staleness ALONE cannot separate a do-nothing
            # backend (NoMemory) from a fresh one — recall/contradiction-accuracy
            # do that. NaiveRAG returns items and so honestly earns staleness>0.
            vals.append(0.0)
        else:
            vals.append(r.superseded_returned / r.retrieved_count)
    return sum(vals) / len(vals)


__all__ = ["staleness"]

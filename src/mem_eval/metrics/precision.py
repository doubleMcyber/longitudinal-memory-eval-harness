"""Metric 2 — precision@k (PRD §7.2): |R_k ∩ G| / |R_k|, averaged over queries.

When nothing is retrieved (|R_k| = 0) the query surfaced nothing relevant, so it
contributes 0 (a memory floor should not get free precision credit for silence).
"""

from __future__ import annotations

from mem_eval.metrics.records import EvalRecord


def precision_at_k(records: list[EvalRecord]) -> float:
    vals = []
    for r in records:
        if r.retrieved_count == 0:
            vals.append(0.0)
            continue
        vals.append(len(r.retrieved_facts & r.gold_support) / r.retrieved_count)
    return sum(vals) / len(vals) if vals else 0.0


__all__ = ["precision_at_k"]

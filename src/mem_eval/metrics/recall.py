"""Metric 1 — recall@k (PRD §7.1): |R_k ∩ G| / |G|, averaged over queries."""

from __future__ import annotations

from mem_eval.metrics.records import EvalRecord


def recall_at_k(records: list[EvalRecord]) -> float:
    vals = []
    for r in records:
        g = r.gold_support
        if not g:
            continue
        vals.append(len(r.retrieved_facts & g) / len(g))
    return sum(vals) / len(vals) if vals else 0.0


__all__ = ["recall_at_k"]

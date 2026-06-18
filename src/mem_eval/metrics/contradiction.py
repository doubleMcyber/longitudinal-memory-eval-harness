"""Metric 3 — contradiction-resolution accuracy (PRD §7.3).

A contradiction query *passes* when the current fact is in R_k AND no superseded
version of it appears in R_k. When the backend returns an answer, the offline
judge additionally requires the answer to reflect the current value (not stale).
Reported as the fraction of contradiction queries that pass.
"""

from __future__ import annotations

from mem_eval.metrics.records import EvalRecord


def query_passes(r: EvalRecord) -> bool:
    """A single contradiction query passes iff the current fact is in R_k, no
    superseded version is in R_k, and (if an answer was synthesized) it reflects
    the current value."""
    rf = r.retrieved_facts
    current_present = r.gold_support <= rf
    stale_present = bool(r.gold_superseded & rf)
    item_pass = current_present and not stale_present
    if not item_pass:
        return False
    if r.answer is not None and r.answer.strip():
        return r.answer_correct
    return True


def resolution_accuracy(records: list[EvalRecord]) -> float:
    probes = [r for r in records if r.gold_superseded]
    if not probes:
        return 0.0
    return sum(1 for r in probes if query_passes(r)) / len(probes)


__all__ = ["resolution_accuracy", "query_passes"]

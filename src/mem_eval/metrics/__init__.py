"""Metric aggregation: assemble all 7 metrics per-category and overall (PRD §7).

Quality metrics (1-4) participate in the determinism hash; operational metrics
(5-7) are reported but excluded (PRD §7, §10.3).
"""

from __future__ import annotations

from mem_eval.data.schema import CATEGORIES
from mem_eval.metrics.contradiction import query_passes, resolution_accuracy
from mem_eval.metrics.cost import cost_per_query
from mem_eval.metrics.latency import latency_percentiles
from mem_eval.metrics.precision import precision_at_k
from mem_eval.metrics.recall import recall_at_k
from mem_eval.metrics.records import EvalRecord
from mem_eval.metrics.significance import bootstrap_ci
from mem_eval.metrics.staleness import staleness
from mem_eval.metrics.storage import storage_growth

# Quality metrics that participate in the determinism hash (PRD §7, §10.3).
QUALITY_KEYS = (
    "recall_at_k",
    "precision_at_k",
    "contradiction_resolution_accuracy",
    "staleness",
)


def _answer_accuracy(records: list[EvalRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.answer_correct) / len(records)


def _quality_vectors(records: list[EvalRecord]) -> dict[str, list[float]]:
    """Per-query value vectors for the four quality metrics, for bootstrap CIs."""
    recall_v = [
        len(r.retrieved_facts & r.gold_support) / len(r.gold_support)
        for r in records
        if r.gold_support
    ]
    precision_v = [
        (len(r.retrieved_facts & r.gold_support) / r.retrieved_count) if r.retrieved_count else 0.0
        for r in records
    ]
    probes = [r for r in records if r.gold_superseded]
    contradiction_v = [1.0 if query_passes(r) else 0.0 for r in probes]
    staleness_v = [
        (r.superseded_returned / r.retrieved_count) if r.retrieved_count else 0.0 for r in probes
    ]
    return {
        "recall_at_k": recall_v,
        "precision_at_k": precision_v,
        "contradiction_resolution_accuracy": contradiction_v,
        "staleness": staleness_v,
    }


def _intervals(records: list[EvalRecord]) -> dict:
    return {k: bootstrap_ci(v) for k, v in _quality_vectors(records).items()}


def _block(records: list[EvalRecord], ingest_usd: float, storage: dict) -> dict:
    cost = cost_per_query(records, ingest_usd)
    return {
        # quality metrics (1-4)
        "recall_at_k": recall_at_k(records),
        "precision_at_k": precision_at_k(records),
        "contradiction_resolution_accuracy": resolution_accuracy(records),
        "staleness": staleness(records),
        # operational metrics (5-7)
        "latency_ms": latency_percentiles(records),
        "cost_per_query_usd": cost["query_usd"],
        "storage": storage,
        # supporting detail
        "cost": cost,
        "answer_accuracy": _answer_accuracy(records),
        "num_queries": len(records),
        # statistical rigor: bootstrap 95% CIs on the quality metrics (reported,
        # excluded from the determinism hash).
        "ci95": _intervals(records),
    }


def compute_metrics_block(
    records: list[EvalRecord],
    ingest_usd: float,
    storage: dict,
) -> dict:
    """overall + by_category blocks. Per-category amortized ingest uses the same
    global per-query denominator so the figure is comparable across categories."""
    n_all = len(records) or 1
    overall = _block(records, ingest_usd, storage)
    by_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        recs = [r for r in records if r.category == cat]
        cat_ingest = ingest_usd * (len(recs) / n_all)  # keeps per-query amortization constant
        by_category[cat] = _block(recs, cat_ingest, storage)
    return {"overall": overall, "by_category": by_category}


__all__ = [
    "compute_metrics_block",
    "storage_growth",
    "QUALITY_KEYS",
    "recall_at_k",
    "precision_at_k",
    "resolution_accuracy",
    "staleness",
    "latency_percentiles",
    "cost_per_query",
]

"""Metric 6 — cost per query (PRD §7.6).

USD and tokens per query at query time, with amortized ingest cost
(ingest_usd / num_queries) reported separately. Operational metric: excluded
from the determinism hash.
"""

from __future__ import annotations

from mem_eval.metrics.records import EvalRecord


def cost_per_query(records: list[EvalRecord], ingest_usd: float = 0.0) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {
            "query_usd": 0.0,
            "query_tokens": 0.0,
            "amortized_ingest_usd": 0.0,
        }
    total_usd = sum(r.usage.usd for r in records)
    total_tokens = sum(
        r.usage.prompt_tokens + r.usage.completion_tokens + r.usage.embedding_tokens
        for r in records
    )
    return {
        "query_usd": total_usd / n,
        "query_tokens": total_tokens / n,
        "amortized_ingest_usd": ingest_usd / n,
    }


__all__ = ["cost_per_query"]

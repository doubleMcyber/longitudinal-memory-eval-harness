"""Curation-quality tests (PRD §1, §7.7) — Wave 7.

A curating backend (real consolidate()) must keep storage growth flat under
redundant re-assertion, while an uncurated one grows — and curation must be
information-preserving (no loss of the canonical fact or time discipline).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from mem_eval.adapters import TemporalRAG, get_backend
from mem_eval.adapters.base import Session, Turn
from mem_eval.data.generators import curation_stress
from mem_eval.data.schema import Suite
from mem_eval.data.suite import _category_rng
from mem_eval.runner.orchestrate import run_eval


def _curation_suite(seed=42, noise=40) -> Suite:
    scs = curation_stress.generate(_category_rng(seed, 9), count=1, noise_sessions=noise)
    return Suite("v1", "curation", seed, scs)


def test_consolidate_dedups_exact_duplicates_keeping_earliest():
    b = TemporalRAG()
    b.reset()
    for i in range(5):
        b.ingest(Session(f"s{i}", datetime(2025, 1, 1) + timedelta(days=i),
                         [Turn(f"s{i}-t0", "user", "the wifi password is hunter2")]))
    assert b.stats().item_count == 5
    usage = b.consolidate()
    assert b.stats().item_count == 1  # collapsed to a single copy
    assert usage.prompt_tokens == 4   # 4 duplicates removed
    # the retained copy is the EARLIEST (time-discipline safe)
    res = b.query("wifi password", 5, datetime(2025, 1, 10))
    assert res.items and res.items[0].source_session == "s0"


def test_naive_consolidate_is_noop():
    nb = get_backend("naive_rag")
    nb.reset()
    for i in range(5):
        nb.ingest(Session(f"s{i}", datetime(2025, 1, 1) + timedelta(days=i),
                          [Turn(f"s{i}-t0", "user", "the wifi password is hunter2")]))
    nb.consolidate()
    assert nb.stats().item_count == 5  # no curation


def test_curation_flattens_storage_growth():
    suite = _curation_suite(noise=40)
    naive = run_eval(get_backend("naive_rag"), suite, k=10,
                     consolidate_cadence="per_session")["metrics"]["overall"]["storage"]
    temporal = run_eval(get_backend("temporal_rag"), suite, k=10,
                        consolidate_cadence="per_session")["metrics"]["overall"]["storage"]
    # uncurated store grows with redundant sessions; curated store stays compact
    assert naive["growth_slope"] > 0
    assert temporal["growth_slope"] < naive["growth_slope"]
    # and the curated footprint at the deepest sample is far smaller
    naive_bytes = max(int(v) for v in naive["bytes_at_n"].values())
    temporal_bytes = max(int(v) for v in temporal["bytes_at_n"].values())
    assert temporal_bytes < naive_bytes


def test_curation_preserves_recall_of_canonical_fact():
    """Deduplication must not drop the gold fact."""
    suite = _curation_suite(noise=30)
    temporal = run_eval(get_backend("temporal_rag"), suite, k=10)["metrics"]["overall"]
    assert temporal["recall_at_k"] == 1.0

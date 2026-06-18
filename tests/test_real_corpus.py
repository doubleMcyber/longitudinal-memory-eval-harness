"""Meaningfully-sized real-log corpus + multi-scenario importer + IAA (D3).

The corpus is a hand-authored, real-STYLE stand-in for human-collected logs. These
tests pin that it (a) is non-trivially sized, (b) carries hand-labeled gold that
resolves to real turns, (c) runs end-to-end through the SAME metrics as synthetic
suites, (d) is deterministic, and (e) clears an inter-annotator agreement bar.
"""

from __future__ import annotations

from mem_eval.adapters import get_backend
from mem_eval.data.importers.transcript import (
    build_suite_from_corpus,
    corpus_candidate_refs,
    corpus_iaa,
)
from mem_eval.data.schema import CONTRADICTION, suite_content_hash
from mem_eval.runner.orchestrate import run_eval

_QUALITY_KEYS = (
    "recall_at_k", "precision_at_k", "contradiction_resolution_accuracy", "staleness",
    "latency_ms", "cost_per_query_usd", "storage",
)


def test_corpus_is_meaningfully_sized():
    suite = build_suite_from_corpus()
    n_scen = len(suite.scenarios)
    n_queries = sum(len(s.queries) for s in suite.scenarios)
    n_sessions = sum(len(s.sessions) for s in suite.scenarios)
    # a real step up from the single 3-session / 1-query sample fixture
    assert n_scen >= 5
    assert n_queries >= 8
    assert n_sessions >= 20
    # spans multiple capability categories (not one repeated shape)
    assert len({s.category for s in suite.scenarios}) >= 4


def test_every_gold_ref_resolves_to_a_real_turn():
    suite = build_suite_from_corpus()
    for s in suite.scenarios:
        valid = set(s.turn_to_fact.values())
        for q in s.queries:
            assert q.gold_support, f"{q.query_id} has empty support"
            for ref in q.gold_support + q.gold_superseded:
                assert ref in valid, f"{q.query_id}: {ref} does not resolve to a turn"


def test_corpus_has_contradiction_probes_with_superseded_gold():
    suite = build_suite_from_corpus()
    probes = [q for s in suite.scenarios for q in s.queries if q.gold_superseded]
    assert probes, "corpus must contain contradiction probes (superseded gold)"
    assert any(q.category == CONTRADICTION for q in probes)


def test_corpus_runs_through_same_metrics():
    suite = build_suite_from_corpus()
    sc = run_eval(get_backend("naive_rag"), suite, k=10, timestamp="1970-01-01T00:00:00")
    overall = sc["metrics"]["overall"]
    for key in _QUALITY_KEYS:
        assert key in overall
    assert overall["recall_at_k"] > 0.0
    assert sc["determinism_hash"]


def test_corpus_is_deterministic():
    assert suite_content_hash(build_suite_from_corpus()) == suite_content_hash(build_suite_from_corpus())


def test_candidate_refs_cover_all_turns():
    suite = build_suite_from_corpus()
    refs = corpus_candidate_refs()
    for s in suite.scenarios:
        assert set(refs[s.scenario_id]) == set(s.turn_to_fact.values())


def test_inter_annotator_agreement_high_but_imperfect():
    iaa = corpus_iaa()
    assert iaa["n_annotators"] == 3
    assert iaa["answer_agreement"] == 1.0
    # high agreement on support membership, but a couple of defensible disagreements
    assert 0.6 < iaa["support_kappa"] < 1.0
    assert iaa["n_shared_queries"] >= 5

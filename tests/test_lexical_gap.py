"""Lexical-gap / semantic-retrieval tests (PRD §1, §6.1) — Wave 6.

Proves retrieval measures memory, not string overlap: on scenarios where the
query is a synonym of the fact (no shared content tokens) amid same-entity
distractors, lexical token-cosine fails but a semantic embedding succeeds.
"""

from __future__ import annotations

import pytest

from mem_eval.adapters import ConfigurableRAG, REGISTRY, get_backend
from mem_eval.adapters.configurable_rag import SemanticRAG
from mem_eval.adapters.embeddings import SynonymHashEmbedding, TokenCosineEmbedding
from mem_eval.data.generators import lexical_gap
from mem_eval.data.schema import LONGITUDINAL_RECALL, Suite
from mem_eval.data.suite import _category_rng, build_suite
from mem_eval.runner.orchestrate import run_eval


def _lexgap_suite(seed: int, count: int = 6) -> Suite:
    scs = lexical_gap.generate(_category_rng(seed, 5), count=count)
    return Suite("v1", "lexgap", seed, scs)


@pytest.mark.parametrize("seed", [42, 7, 123, 2024])
def test_semantic_beats_lexical_on_gap(seed):
    suite = _lexgap_suite(seed)
    lexical = run_eval(ConfigurableRAG(TokenCosineEmbedding()), suite, k=10)["metrics"]["overall"]
    semantic = run_eval(ConfigurableRAG(SynonymHashEmbedding()), suite, k=10)["metrics"]["overall"]
    floor = run_eval(get_backend("no_memory"), suite, k=10)["metrics"]["overall"]
    # lexical retrieval is genuinely hard here (it can't bridge the synonym gap)
    assert lexical["recall_at_k"] < 0.5
    # a semantic embedding recovers the target
    assert semantic["recall_at_k"] > lexical["recall_at_k"]
    assert semantic["recall_at_k"] >= 0.9
    assert floor["recall_at_k"] == 0.0


def test_semantic_rag_registered_and_beats_naive_overall():
    assert "semantic_rag" in REGISTRY
    s = build_suite("v1", seed=42)
    naive = run_eval(get_backend("naive_rag"), s, k=10)["metrics"]["overall"]
    semantic = run_eval(get_backend("semantic_rag"), s, k=10)["metrics"]["overall"]
    # semantic wins the lexical-gap longitudinal scenarios -> higher overall recall
    assert semantic["recall_at_k"] > naive["recall_at_k"]


def test_default_configurable_still_matches_naive_with_lexgap_present():
    """The lexical-gap addition must not break the faithful-wiring guarantee."""
    s = build_suite("v1", seed=42)
    naive = run_eval(get_backend("naive_rag"), s, k=10)["metrics"]["overall"]
    conf = run_eval(ConfigurableRAG(), s, k=10)["metrics"]["overall"]
    for key in ("recall_at_k", "precision_at_k", "contradiction_resolution_accuracy", "staleness"):
        assert conf[key] == naive[key]


def test_synonym_embedding_deterministic_dense():
    e = SynonymHashEmbedding(dim=64)
    assert e.embed("alice relocated to berlin") == e.embed("alice relocated to berlin")
    # synonyms of the same concept are more similar than unrelated text
    reside_a = e.embed("alice relocated to berlin")
    reside_q = e.embed("where does alice live now")
    unrelated = e.embed("alice enjoys chess")
    assert e.similarity(reside_a, reside_q) > e.similarity(reside_a, unrelated)


def test_lexgap_scenarios_are_longitudinal_category():
    # they keep the suite at exactly 5 categories (A2) by tagging longitudinal_recall
    for sc in _lexgap_suite(42).scenarios:
        assert sc.category == LONGITUDINAL_RECALL
        assert sc.queries[0].gold_support  # exact structured gold

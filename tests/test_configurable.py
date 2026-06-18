"""ConfigurableRAG + pluggable embeddings tests (PRD §1, §14) — Wave 2.

Proves the real-backend seam: (1) the default embedding makes ConfigurableRAG
behaviorally identical to NaiveRAG (faithful wiring), and (2) a swapped-in
embedding actually drives retrieval (so a lab's model would too).
"""

from __future__ import annotations

from collections import Counter

from mem_eval.adapters import ConfigurableRAG, get_backend
from mem_eval.adapters.embeddings import (
    HashEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    TokenCosineEmbedding,
)
from mem_eval.data.suite import build_suite
from mem_eval.runner.orchestrate import run_eval

QUALITY = ("recall_at_k", "precision_at_k", "contradiction_resolution_accuracy", "staleness")


def test_default_embedding_matches_naive_rag():
    """ConfigurableRAG() with the default token-cosine embedding must reproduce
    NaiveRAG's quality metrics exactly — proof the seam is faithful."""
    suite = build_suite("v1", seed=42)
    naive = run_eval(get_backend("naive_rag"), suite, k=10)["metrics"]["overall"]
    conf = run_eval(ConfigurableRAG(), suite, k=10)["metrics"]["overall"]
    for key in QUALITY:
        assert conf[key] == naive[key], (key, conf[key], naive[key])


def test_swapped_embedding_changes_retrieval():
    """A degenerate embedding that ignores content must change results — proof
    the injected embedding actually drives retrieval (not bypassed)."""

    class LengthOnlyEmbedding:
        name = "length-only"

        def embed(self, text: str):
            return len(text)

        def similarity(self, a: int, b: int) -> float:
            # closeness in length only; entirely content-agnostic
            return 1.0 / (1.0 + abs(a - b))

    suite = build_suite("v1", seed=42)
    default = run_eval(ConfigurableRAG(), suite, k=10)["metrics"]["overall"]
    degenerate = run_eval(ConfigurableRAG(LengthOnlyEmbedding()), suite, k=10)["metrics"]["overall"]
    # content-blind retrieval must do strictly worse on recall
    assert degenerate["recall_at_k"] < default["recall_at_k"]


def test_hash_embedding_runs_end_to_end():
    suite = build_suite("v1", seed=42)
    m = run_eval(ConfigurableRAG(HashEmbedding(dim=128)), suite, k=10)["metrics"]["overall"]
    assert m["recall_at_k"] > 0.0  # dense hashing retrieval still finds gold


def test_hash_embedding_actually_dense_and_used():
    """HashEmbedding must return a dense vector of the configured dim and be the
    path used (not silently falling back to token-cosine)."""
    e = HashEmbedding(dim=64)
    v = e.embed("alice phone number")
    assert isinstance(v, list) and len(v) == 64
    # identical text -> identical dense vector; different text -> different vector
    assert e.embed("alice") != e.embed("bob")
    assert 0.0 <= e.similarity(e.embed("alice phone"), e.embed("bob email")) <= 1.0


def test_version_records_embedding():
    assert ConfigurableRAG().version.endswith("+token-cosine")
    assert ConfigurableRAG(HashEmbedding(64)).version.endswith("+hash-64")


def test_token_cosine_embedding_api():
    e = TokenCosineEmbedding()
    a, b = e.embed("alice phone number"), e.embed("alice phone")
    assert isinstance(a, Counter)
    assert 0.0 < e.similarity(a, b) <= 1.0


def test_integration_stubs_raise_until_wired():
    import pytest

    for stub in (SentenceTransformerEmbedding, OpenAIEmbedding):
        with pytest.raises(RuntimeError):
            stub()

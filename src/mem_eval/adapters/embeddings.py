"""Pluggable embedding functions — the real-backend retrieval path (PRD §1, §14).

A benchmark labs run must let them plug in *their* embedding model. ``EmbeddingFn``
is the seam: anything that can turn text into a vector and score two vectors can
drive retrieval via :class:`~mem_eval.adapters.configurable_rag.ConfigurableRAG`.

Shipped offline (deterministic, no deps):
* ``TokenCosineEmbedding`` — the default; sparse bag-of-tokens + cosine (identical
  behavior to NaiveRAG, so default ConfigurableRAG == NaiveRAG).
* ``HashEmbedding`` — deterministic dense hashing embedding; a real-shaped dense
  vector path that exercises the dense code without any model download.

Documented integration stubs (lazy-import; raise an informative error until the
dependency is installed) — these are the wiring points for production:
* ``SentenceTransformerEmbedding`` — local HF sentence-transformers.
* ``OpenAIEmbedding`` — hosted embeddings API.

Determinism note: the offline embeddings are pure functions of text, so runs stay
reproducible. Real embeddings make the run only as deterministic as the model;
the scorecard records the embedding model id for fair comparison (PRD §10.3).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

from mem_eval.text import cosine, vectorize


@runtime_checkable
class EmbeddingFn(Protocol):
    name: str

    def embed(self, text: str) -> object:
        ...

    def similarity(self, a: object, b: object) -> float:
        ...


class TokenCosineEmbedding:
    """Default offline embedding: sparse term-frequency vector + cosine."""

    name = "token-cosine"

    def embed(self, text: str) -> Counter:
        return vectorize(text)

    def similarity(self, a: Counter, b: Counter) -> float:
        return cosine(a, b)


class HashEmbedding:
    """Deterministic dense hashing embedding (the 'hashing trick'). Tokens are
    hashed into a fixed-dim vector with signed buckets; similarity is dense
    cosine. No model download, fully reproducible — a stand-in shaped like a real
    dense embedding for exercising the dense path."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in vectorize(text):
            h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return vec

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        if dot == 0:
            return 0.0
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0


class SynonymHashEmbedding:
    """Deterministic offline SEMANTIC embedding demonstrator. Like HashEmbedding,
    but additionally hashes a concept bucket for any token in a synonym lexicon,
    so paraphrases of the same relation collide on a shared dimension. This bridges
    the lexical gap that defeats token-cosine — demonstrating, fully offline and
    reproducibly, that a better embedding beats lexical retrieval (the production
    signal comes from real models plugged through the same EmbeddingFn seam).

    The lexicon is generic world knowledge (synonym sets), not gold."""

    def __init__(self, lexicon: dict[str, str] | None = None, dim: int = 256,
                 concept_weight: float = 2.0) -> None:
        if lexicon is None:
            from mem_eval.data.paraphrase import TOKEN_CONCEPT

            lexicon = TOKEN_CONCEPT
        self.lexicon = lexicon
        self.dim = dim
        self.concept_weight = concept_weight
        self.name = f"synonym-hash-{dim}"

    @staticmethod
    def _bucket(token: str, dim: int):
        h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        return h % dim, (1.0 if (h >> 8) & 1 else -1.0)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in vectorize(text):
            idx, sign = self._bucket(tok, self.dim)
            vec[idx] += sign
            concept = self.lexicon.get(tok)
            if concept:
                cidx, csign = self._bucket(f"CONCEPT:{concept}", self.dim)
                vec[cidx] += csign * self.concept_weight
        return vec

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        if dot == 0:
            return 0.0
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0


class _LazyEmbeddingStub:
    """Base for integration stubs that need an optional dependency."""

    _import_hint = ""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(self._import_hint)


class SentenceTransformerEmbedding(_LazyEmbeddingStub):
    """STUB: wrap a local sentence-transformers model.

    Production wiring (once `sentence-transformers` is installed)::

        from sentence_transformers import SentenceTransformer
        class STEmbedding:
            name = "all-MiniLM-L6-v2"
            def __init__(self): self.m = SentenceTransformer(self.name)
            def embed(self, text): return self.m.encode(text)
            def similarity(self, a, b): return float(util.cos_sim(a, b))
    """

    _import_hint = (
        "SentenceTransformerEmbedding is a documented integration stub. Install "
        "`sentence-transformers` and implement embed/similarity (see docstring)."
    )


class OpenAIEmbedding(_LazyEmbeddingStub):
    """STUB: wrap a hosted embeddings API (records model id in the scorecard)."""

    _import_hint = (
        "OpenAIEmbedding is a documented integration stub. Provide an API client "
        "and implement embed/similarity returning dense vectors + cosine."
    )


__all__ = [
    "EmbeddingFn",
    "TokenCosineEmbedding",
    "HashEmbedding",
    "SynonymHashEmbedding",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
]

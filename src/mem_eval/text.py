"""Deterministic, offline text utilities.

The harness must be reproducible with no network/LLM dependency (PRD §10.3,
§14). NaiveRAG's "embedding" is a sparse bag-of-tokens vector and similarity is
exact cosine over those vectors — fully deterministic and inspectable. Token
counts drive synthetic usage/cost accounting.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal stopword set: function words carry no retrieval signal and would
# otherwise inflate cosine between unrelated sentences.
_STOPWORDS = frozenset(
    """
    a an the of to in on at for and or but is are was were be been being
    do does did has have had will would can could should i you he she it we they
    my your his her its our their me him them this that these those as by with
    about into over under again then once here there all any both each more most
    what which who whom whose when where why how than too very s t just
    """.split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumeric runs, optionally drop stopwords."""
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def vectorize(text: str) -> Counter:
    """Sparse term-frequency vector (the offline 'embedding')."""
    return Counter(tokenize(text))


def cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity between two sparse vectors. 0 when either is empty."""
    if not a or not b:
        return 0.0
    # iterate the smaller for the dot product
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(w * large.get(tok, 0) for tok, w in small.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb)


def count_tokens(text: str) -> int:
    """Token count for usage accounting (raw, includes stopwords ~ realistic)."""
    return len(_TOKEN_RE.findall(text.lower()))


def byte_size(text: str) -> int:
    return len(text.encode("utf-8"))


__all__ = ["tokenize", "vectorize", "cosine", "count_tokens", "byte_size"]

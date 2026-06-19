"""Backend registry. Adding a backend = implement §4 and register here."""

from __future__ import annotations

from mem_eval.adapters.base import (
    BackendStats,
    BaseBackend,
    MemoryBackend,
    MemoryItem,
    QueryResult,
    Session,
    Turn,
    Usage,
)
from mem_eval.adapters.configurable_rag import ConfigurableRAG, SemanticRAG
from mem_eval.adapters.curated_brain import CuratedBrain
from mem_eval.adapters.letta import Letta
from mem_eval.adapters.long_context import LongContext
from mem_eval.adapters.naive_rag import NaiveRAG
from mem_eval.adapters.no_memory import NoMemory
from mem_eval.adapters.temporal_rag import TemporalRAG  # noqa: F401 (re-exported)

# Runnable baselines shipped in-box (PRD §5): floor + naive ceiling.
BASELINES: dict[str, type] = {
    NoMemory.name: NoMemory,
    LongContext.name: LongContext,
    NaiveRAG.name: NaiveRAG,
}

# Reference backends: genuinely curated systems the baselines must be beaten by.
# These make the scoreboard a meaningful spectrum (floor -> naive -> good), so a
# real memory system has a credible target to clear.
REFERENCES: dict[str, type] = {
    TemporalRAG.name: TemporalRAG,
}

# Integration backends: real-stack seams. ConfigurableRAG takes an injected
# EmbeddingFn (default offline == NaiveRAG; plug in a real embedding model).
INTEGRATIONS: dict[str, type] = {
    ConfigurableRAG.name: ConfigurableRAG,
    SemanticRAG.name: SemanticRAG,
}

# The system-under-test: Curated Brain, wired to the `curated-brain` package.
SYSTEM: dict[str, type] = {
    CuratedBrain.name: CuratedBrain,
}

# Documented stubs (interface-complete, raise NotImplementedError until wired).
STUBS: dict[str, type] = {
    Letta.name: Letta,
}

REGISTRY: dict[str, type] = {**BASELINES, **REFERENCES, **INTEGRATIONS, **SYSTEM, **STUBS}


def get_backend(name: str) -> MemoryBackend:
    """Instantiate a backend by registry name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()  # type: ignore[return-value]


__all__ = [
    "BASELINES",
    "REFERENCES",
    "INTEGRATIONS",
    "STUBS",
    "REGISTRY",
    "get_backend",
    "NoMemory",
    "LongContext",
    "NaiveRAG",
    "TemporalRAG",
    "ConfigurableRAG",
    "SemanticRAG",
    "Letta",
    "CuratedBrain",
    "MemoryBackend",
    "BaseBackend",
    "Turn",
    "Session",
    "Usage",
    "MemoryItem",
    "QueryResult",
    "BackendStats",
]

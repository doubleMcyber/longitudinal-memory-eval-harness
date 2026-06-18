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
from mem_eval.adapters.curated_brain import CuratedBrain
from mem_eval.adapters.letta import Letta
from mem_eval.adapters.long_context import LongContext
from mem_eval.adapters.naive_rag import NaiveRAG
from mem_eval.adapters.no_memory import NoMemory

# Runnable baselines shipped in-box (PRD §5).
BASELINES: dict[str, type] = {
    NoMemory.name: NoMemory,
    LongContext.name: LongContext,
    NaiveRAG.name: NaiveRAG,
}

# Documented stubs (interface-complete, raise NotImplementedError until wired).
STUBS: dict[str, type] = {
    Letta.name: Letta,
    CuratedBrain.name: CuratedBrain,
}

REGISTRY: dict[str, type] = {**BASELINES, **STUBS}


def get_backend(name: str) -> MemoryBackend:
    """Instantiate a backend by registry name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()  # type: ignore[return-value]


__all__ = [
    "BASELINES",
    "STUBS",
    "REGISTRY",
    "get_backend",
    "NoMemory",
    "LongContext",
    "NaiveRAG",
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

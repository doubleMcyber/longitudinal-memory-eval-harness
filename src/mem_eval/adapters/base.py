"""The backend adapter interface — the central, frozen contract (PRD §4).

Everything in the harness depends on this module. It is frozen as of Stage 1:
adding a backend means implementing :class:`MemoryBackend` and nothing else.
The harness never special-cases a backend.

Contract rules (PRD §4.3), enforced by the acceptance suite:

1. Time discipline   — ``query(as_of=T)`` only uses sessions with ``timestamp <= T``.
2. Provenance        — every returned ``MemoryItem`` carries ``source_session`` and
                        ``timestamp``; ``item_id`` is stable within a run.
3. Isolation         — ``reset()`` fully clears state.
4. Determinism       — stable results for the same ingest order/config/as_of, or a
                        declared ``stats().extra["nondeterministic"] = True``.
5. Honest accounting — real ``Usage`` from ingest/query/consolidate; harness wall-clocks latency.
6. No gold access    — a backend receives only ``prompt``, ``k``, ``as_of``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# 4.1 Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    turn_id: str
    role: str  # "user" | "assistant" | "system"
    text: str


@dataclass(frozen=True)
class Session:
    session_id: str
    timestamp: datetime  # when this session occurred
    turns: list[Turn]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            usd=self.usd + other.usd,
        )


@dataclass
class MemoryItem:
    item_id: str  # stable id the harness can grade against gold support
    content: str
    source_session: str
    source_turn: str | None
    timestamp: datetime  # provenance: when the underlying info was first seen
    score: float | None = None  # backend's own relevance score (optional)
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    items: list[MemoryItem]  # ranked best-first; graded against gold support
    answer: str | None  # optional synthesized answer; graded when present
    usage: Usage
    latency_ms: float


@dataclass
class BackendStats:
    item_count: int
    bytes: int  # storage footprint; basis for storage-growth metric
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4.2 The interface
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryBackend(Protocol):
    name: str
    version: str  # bumped on any behavior change; recorded in every scorecard

    def reset(self) -> None:
        """Wipe all state; harness calls this before each scenario for isolation."""
        ...

    def ingest(self, session: Session) -> Usage:
        """Persist one session into memory. Called in timestamp order."""
        ...

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        """Retrieve up to k items relevant to prompt, usable for answering."""
        ...

    def consolidate(self) -> Usage:
        """Optional background curation/compaction hook; default no-op."""
        ...

    def stats(self) -> BackendStats:
        """Current storage footprint and item count."""
        ...


# ---------------------------------------------------------------------------
# Shared base implementation: handles boilerplate every concrete backend needs
# without special-casing behavior. Concrete backends override ingest/query.
# ---------------------------------------------------------------------------


class BaseBackend:
    """Convenience base providing default ``consolidate`` (no-op) and a uniform
    ``__init__`` for ``name``/``version``. Concrete backends still implement the
    contract; this only removes duplicated scaffolding."""

    name: str = "base"
    version: str = "0.0.0"

    def reset(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def ingest(self, session: Session) -> Usage:  # pragma: no cover - overridden
        raise NotImplementedError

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:  # pragma: no cover
        raise NotImplementedError

    def consolidate(self) -> Usage:
        # Default: no-op curation hook (PRD §4.2).
        return Usage()

    def stats(self) -> BackendStats:  # pragma: no cover - overridden
        raise NotImplementedError


__all__ = [
    "Turn",
    "Session",
    "Usage",
    "MemoryItem",
    "QueryResult",
    "BackendStats",
    "MemoryBackend",
    "BaseBackend",
]

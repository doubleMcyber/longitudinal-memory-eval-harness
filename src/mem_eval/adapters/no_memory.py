"""NoMemory baseline — the floor (PRD §5).

``ingest`` is a no-op; ``query`` returns nothing and a "don't know" answer.
Any real memory system must beat this; if it does not, the harness or the
backend is broken.
"""

from __future__ import annotations

from datetime import datetime

from mem_eval.adapters.base import (
    BackendStats,
    BaseBackend,
    QueryResult,
    Session,
    Usage,
)
from mem_eval.text import count_tokens

DONT_KNOW = "I don't know."


class NoMemory(BaseBackend):
    name = "no_memory"
    version = "1.0.0"

    def __init__(self) -> None:
        self._ingested = 0

    def reset(self) -> None:
        self._ingested = 0

    def ingest(self, session: Session) -> Usage:
        # No-op: nothing is stored. Honest accounting: zero work.
        return Usage()

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        # Returns no items and a refusal. Prompt is still "read" (counted).
        return QueryResult(
            items=[],
            answer=DONT_KNOW,
            usage=Usage(prompt_tokens=count_tokens(prompt)),
            latency_ms=0.0,
        )

    def stats(self) -> BackendStats:
        return BackendStats(item_count=0, bytes=0)


__all__ = ["NoMemory", "DONT_KNOW"]

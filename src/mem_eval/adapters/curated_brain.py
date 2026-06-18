"""CuratedBrain adapter — STUB (PRD §5, §2 non-goals).

The system-under-test: the backend the scoreboard ultimately judges. v1 does
NOT build it — the harness exists to *measure* it. Shipped interface-complete
so it slots into the registry the moment it is implemented; until then every
REQUIRED operation (reset/ingest/query/stats) raises NotImplementedError.
``consolidate`` inherits the contract's default no-op hook (PRD §4.2).
"""

from __future__ import annotations

from datetime import datetime

from mem_eval.adapters.base import BackendStats, BaseBackend, QueryResult, Session, Usage

_MSG = (
    "CuratedBrain is the system-under-test and ships as a stub in v1 (PRD §2 non-goals). "
    "Implement §4 to make it judgeable by the harness."
)


class CuratedBrain(BaseBackend):
    name = "curated_brain"
    version = "0.0.0-stub"

    def reset(self) -> None:
        raise NotImplementedError(_MSG)

    def ingest(self, session: Session) -> Usage:
        raise NotImplementedError(_MSG)

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        raise NotImplementedError(_MSG)

    def stats(self) -> BackendStats:
        raise NotImplementedError(_MSG)


__all__ = ["CuratedBrain"]

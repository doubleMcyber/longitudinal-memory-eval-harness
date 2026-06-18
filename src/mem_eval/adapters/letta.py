"""Letta adapter — STUB (PRD §5).

The canonical "external backend" integration path. Interface-complete so it can
be registered and type-checked, but every REQUIRED operation
(reset/ingest/query/stats) raises NotImplementedError until the third-party
system is wired in. ``consolidate`` inherits the contract's default no-op hook
(PRD §4.2). This documents exactly what a new external backend must implement:
§4 and nothing else.
"""

from __future__ import annotations

from datetime import datetime

from mem_eval.adapters.base import BackendStats, BaseBackend, QueryResult, Session, Usage

_MSG = "Letta backend is a documented stub (PRD §5); wire the external system to enable it."


class Letta(BaseBackend):
    name = "letta"
    version = "0.0.0-stub"

    def reset(self) -> None:
        raise NotImplementedError(_MSG)

    def ingest(self, session: Session) -> Usage:
        raise NotImplementedError(_MSG)

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        raise NotImplementedError(_MSG)

    def stats(self) -> BackendStats:
        raise NotImplementedError(_MSG)


__all__ = ["Letta"]

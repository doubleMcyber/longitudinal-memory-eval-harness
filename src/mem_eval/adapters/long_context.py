"""LongContext baseline — "just stuff the window" (PRD §5).

Stores raw sessions. ``query`` concatenates all sessions with
``timestamp <= as_of`` most-recent-first up to a token budget ``B``, then
answers from that window. No retrieval index, no ranking. Exposes cost/latency
blow-up and storage growth, and hits a ceiling once history exceeds ``B``.
"""

from __future__ import annotations

from datetime import datetime

from mem_eval.adapters.base import (
    BackendStats,
    BaseBackend,
    MemoryItem,
    QueryResult,
    Session,
    Usage,
)
from mem_eval.adapters.no_memory import DONT_KNOW
from mem_eval.pricing import price
from mem_eval.text import byte_size, cosine, count_tokens, vectorize

DEFAULT_TOKEN_BUDGET = 1500


class _StoredTurn:
    __slots__ = ("session_id", "turn_id", "text", "timestamp", "tokens", "vec")

    def __init__(self, session_id: str, turn_id: str, text: str, timestamp: datetime) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.text = text
        self.timestamp = timestamp
        self.tokens = count_tokens(text)
        self.vec = vectorize(text)


class LongContext(BaseBackend):
    name = "long_context"
    version = "1.0.0"

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.token_budget = token_budget
        self._turns: list[_StoredTurn] = []

    def reset(self) -> None:
        self._turns = []

    def ingest(self, session: Session) -> Usage:
        for t in session.turns:
            self._turns.append(_StoredTurn(session.session_id, t.turn_id, t.text, session.timestamp))
        # Storing raw text is cheap compute; cost shows up at query time.
        return Usage()

    def _window(self, as_of: datetime) -> list[_StoredTurn]:
        """Most-recent-first turns with timestamp <= as_of, up to the budget."""
        visible = [t for t in self._turns if t.timestamp <= as_of]
        visible.sort(key=lambda t: (t.timestamp, t.turn_id), reverse=True)
        out: list[_StoredTurn] = []
        spent = 0
        for t in visible:
            if spent + t.tokens > self.token_budget:
                break
            out.append(t)
            spent += t.tokens
        return out

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        window = self._window(as_of)
        # Items: the stuffed window, most-recent-first, capped at k (no ranking).
        items = [
            MemoryItem(
                item_id=f"{t.session_id}:{t.turn_id}",
                content=t.text,
                source_session=t.session_id,
                source_turn=t.turn_id,
                timestamp=t.timestamp,
            )
            for t in window[:k]
        ]
        # Answer: the window's best match for the prompt (the "LLM reads the
        # window" step, offline & deterministic). Found only if within budget B.
        qv = vectorize(prompt)
        best, best_sim = None, 0.0
        for t in window:
            sim = cosine(qv, t.vec)
            if sim > best_sim:
                best, best_sim = t, sim
        answer = best.text if best is not None and best_sim > 0 else DONT_KNOW

        stuffed_tokens = sum(t.tokens for t in window) + count_tokens(prompt)
        completion_tokens = count_tokens(answer)
        usage = Usage(
            prompt_tokens=stuffed_tokens,
            completion_tokens=completion_tokens,
            usd=price(prompt_tokens=stuffed_tokens, completion_tokens=completion_tokens),
        )
        return QueryResult(items=items, answer=answer, usage=usage, latency_ms=0.0)

    def stats(self) -> BackendStats:
        return BackendStats(
            item_count=len(self._turns),
            bytes=sum(byte_size(t.text) for t in self._turns),
            extra={"token_budget": self.token_budget},
        )


__all__ = ["LongContext", "DEFAULT_TOKEN_BUDGET"]

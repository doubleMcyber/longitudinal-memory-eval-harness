"""NaiveRAG baseline — the standard reference (PRD §5).

Chunk sessions (one chunk per turn) -> embed -> store. ``query`` embeds the
prompt and returns top-k by cosine. No contradiction handling and no curation:
it cannot prefer fresh facts over stale ones, so it MUST exhibit
``staleness > 0`` on the contradiction suite (PRD §5, A4).

Ranking is fully deterministic: cosine desc, then recency (timestamp desc),
then item_id asc. The recency tie-break is an honest "naive" choice and is the
mechanism by which an *old* superseded fact sinks below same-similarity
distractors — giving a contradiction-accuracy strictly between the NoMemory
floor and a contradiction-aware backend.
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


class _Chunk:
    __slots__ = ("item_id", "session_id", "turn_id", "text", "timestamp", "vec", "ord")

    def __init__(self, item_id, session_id, turn_id, text, timestamp, ordinal) -> None:
        self.item_id = item_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.text = text
        self.timestamp = timestamp
        self.vec = vectorize(text)
        self.ord = ordinal  # ingest order, for stable tie-break


class NaiveRAG(BaseBackend):
    name = "naive_rag"
    version = "1.0.0"

    def __init__(self) -> None:
        self._chunks: list[_Chunk] = []
        self._n = 0

    def reset(self) -> None:
        self._chunks = []
        self._n = 0

    def ingest(self, session: Session) -> Usage:
        emb_tokens = 0
        for t in session.turns:
            item_id = f"{session.session_id}:{t.turn_id}"
            self._chunks.append(
                _Chunk(item_id, session.session_id, t.turn_id, t.text, session.timestamp, self._n)
            )
            self._n += 1
            emb_tokens += count_tokens(t.text)
        return Usage(embedding_tokens=emb_tokens, usd=price(embedding_tokens=emb_tokens))

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        qv = vectorize(prompt)
        # TIME DISCIPLINE (PRD §4.3.1): only chunks at or before as_of are visible.
        candidates = [c for c in self._chunks if c.timestamp <= as_of]
        scored = [(cosine(qv, c.vec), c) for c in candidates]
        scored = [(s, c) for s, c in scored if s > 0.0]
        # Deterministic ranking: similarity desc, recency desc, item_id asc.
        scored.sort(key=lambda sc: (-sc[0], -sc[1].timestamp.timestamp(), sc[1].item_id))
        top = scored[:k]

        items = [
            MemoryItem(
                item_id=c.item_id,
                content=c.text,
                source_session=c.session_id,
                source_turn=c.turn_id,
                timestamp=c.timestamp,
                score=s,
            )
            for s, c in top
        ]
        answer = items[0].content if items else DONT_KNOW

        emb_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(answer)
        usage = Usage(
            embedding_tokens=emb_tokens,
            completion_tokens=completion_tokens,
            usd=price(embedding_tokens=emb_tokens, completion_tokens=completion_tokens),
        )
        return QueryResult(items=items, answer=answer, usage=usage, latency_ms=0.0)

    def stats(self) -> BackendStats:
        return BackendStats(
            item_count=len(self._chunks),
            bytes=sum(byte_size(c.text) for c in self._chunks),
        )


__all__ = ["NaiveRAG"]

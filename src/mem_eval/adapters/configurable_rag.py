"""ConfigurableRAG — the real-backend integration path (PRD §1, §14).

Same retrieval shape as NaiveRAG, but the embedding is INJECTED via an
``EmbeddingFn``. Plug in sentence-transformers, a hosted embeddings API, or any
custom model and you get a graded scorecard with no other changes — this is the
seam that lets a lab benchmark *their* retrieval stack against the references.

With the default :class:`TokenCosineEmbedding` it is behaviorally identical to
NaiveRAG (same scores), which is exactly how we prove the wiring is faithful.
The embedding model id is folded into ``version`` and ``stats`` so comparisons
hold the embedding constant (PRD §10.3).
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
from mem_eval.adapters.embeddings import EmbeddingFn, TokenCosineEmbedding
from mem_eval.adapters.no_memory import DONT_KNOW
from mem_eval.pricing import price
from mem_eval.text import byte_size, count_tokens


class _Chunk:
    __slots__ = ("item_id", "session_id", "turn_id", "text", "timestamp", "vec")

    def __init__(self, item_id, session_id, turn_id, text, timestamp, vec) -> None:
        self.item_id = item_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.text = text
        self.timestamp = timestamp
        self.vec = vec


class ConfigurableRAG(BaseBackend):
    name = "configurable_rag"

    def __init__(self, embedding: EmbeddingFn | None = None) -> None:
        self.embedding = embedding or TokenCosineEmbedding()
        self.version = f"1.0.0+{self.embedding.name}"
        self._chunks: list[_Chunk] = []

    def reset(self) -> None:
        self._chunks = []

    def ingest(self, session: Session) -> Usage:
        emb_tokens = 0
        for t in session.turns:
            item_id = f"{session.session_id}:{t.turn_id}"
            self._chunks.append(
                _Chunk(item_id, session.session_id, t.turn_id, t.text, session.timestamp,
                       self.embedding.embed(t.text))
            )
            emb_tokens += count_tokens(t.text)
        return Usage(embedding_tokens=emb_tokens, usd=price(embedding_tokens=emb_tokens))

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        qv = self.embedding.embed(prompt)
        candidates = [c for c in self._chunks if c.timestamp <= as_of]
        scored = [(self.embedding.similarity(qv, c.vec), c) for c in candidates]
        scored = [(s, c) for s, c in scored if s > 0.0]
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
        return QueryResult(
            items=items,
            answer=answer,
            usage=Usage(
                embedding_tokens=emb_tokens,
                completion_tokens=completion_tokens,
                usd=price(embedding_tokens=emb_tokens, completion_tokens=completion_tokens),
            ),
            latency_ms=0.0,
        )

    def stats(self) -> BackendStats:
        return BackendStats(
            item_count=len(self._chunks),
            bytes=sum(byte_size(c.text) for c in self._chunks),
            extra={"embedding": self.embedding.name},
        )


__all__ = ["ConfigurableRAG"]

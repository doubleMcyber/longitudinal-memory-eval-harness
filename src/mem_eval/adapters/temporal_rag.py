"""TemporalRAG — a contradiction-aware reference backend (PRD §6.2, §5 spirit).

The three shipped baselines establish a floor (NoMemory) and a naive ceiling
(LongContext/NaiveRAG). A benchmark labs trust must also REWARD good behavior,
so the harness ships one reference backend that genuinely curates memory — and
must therefore score strictly *better* than NaiveRAG on the contradiction and
recency axes while never touching gold labels.

TemporalRAG = NaiveRAG retrieval + two honest, text-only curation mechanisms:

1. Entity-attribute consolidation. From each chunk's surface text it derives a
   *topic key* — the content tokens minus temporal markers (current/previous/…).
   Among retrieved candidates that share a topic key, only the most recent
   assertion survives; older same-topic assertions are treated as superseded and
   dropped. This is exactly what a curated memory does, and it drives staleness
   toward 0 and contradiction-resolution toward 1.
2. Query-cue recency awareness. If the prompt asks for the *current* value
   (current/now/latest/…) the ranker boosts recency; if it asks what was said
   *originally* (originally/first/back then/…) it does not. This lets it satisfy
   intent-dependent recency-vs-relevance queries that pure cosine gets wrong.

Crucially: it sees only (prompt, k, as_of) — no gold, no category, no answer.
The consolidation key is parsed from the same rendered text every backend sees.
"""

from __future__ import annotations

import re
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
from mem_eval.text import byte_size, cosine, count_tokens, tokenize, vectorize

# Temporal markers stripped when forming a topic key (so 'current X' and
# 'previous X' collapse to the same topic and the newer one wins).
_TEMPORAL_MARKERS = frozenset(
    "current previous old new latest former currently now recent earlier "
    "originally first prior updated nowadays presently".split()
)
_RECENCY_CUES = frozenset("current now latest currently today recent nowadays presently".split())
_HISTORY_CUES = frozenset("originally first back then prior earlier former previously initially".split())
_COPULA_RE = re.compile(r"\b(?:is|was|are|were|will)\b")
# subject (possessive owner) + attribute phrase, robust to leading adverbial
# clauses ("After that, Alice's phone number was X").
_POSSESSIVE_RE = re.compile(r"([A-Za-z]+)'s\s+(.*?)\s+(?:is|was|are|were|will)\b", re.I)


def topic_key(text: str) -> frozenset:
    """Subject+attribute signature parsed from surface text, temporal markers
    removed, so every assertion about the same (subject, attribute) collapses to
    one topic regardless of phrasing/recency wording:
    'Alice's current phone number is X', 'Earlier, Alice's phone number was Y',
    and 'After that, Alice's phone number was Z' all -> {alice, phone, number};
    'Bob's current phone number ...' differs. Falls back to the copula-left side
    for non-possessive forms like 'The vault passcode is X' -> {vault, passcode}."""
    m = _POSSESSIVE_RE.search(text)
    if m:
        subject = m.group(1).lower()
        attr_toks = [t for t in tokenize(m.group(2)) if t not in _TEMPORAL_MARKERS]
        return frozenset([subject, *attr_toks])
    left = _COPULA_RE.split(text, maxsplit=1)[0]
    toks = [t for t in tokenize(left) if t not in _TEMPORAL_MARKERS]
    return frozenset(toks)


class _Chunk:
    __slots__ = ("item_id", "session_id", "turn_id", "text", "timestamp", "vec", "topic", "ord")

    def __init__(self, item_id, session_id, turn_id, text, timestamp, ordinal) -> None:
        self.item_id = item_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.text = text
        self.timestamp = timestamp
        self.vec = vectorize(text)
        self.topic = topic_key(text)
        self.ord = ordinal


class TemporalRAG(BaseBackend):
    name = "temporal_rag"
    version = "1.0.0"

    # in 'recency' mode, items scoring at least this fraction of the best cosine
    # are considered "relevant enough" and then ordered by recency.
    RELEVANT_FRAC = 0.5

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

    def _cue_mode(self, prompt: str) -> str:
        toks = set(tokenize(prompt))
        if toks & _RECENCY_CUES:
            return "recency"
        if toks & _HISTORY_CUES:
            return "history"
        return "neutral"

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        qv = vectorize(prompt)
        mode = self._cue_mode(prompt)
        candidates = [c for c in self._chunks if c.timestamp <= as_of]
        scored = [(cosine(qv, c.vec), c) for c in candidates]
        scored = [(s, c) for s, c in scored if s > 0.0]
        if not scored:
            return QueryResult(
                items=[],
                answer=DONT_KNOW,
                usage=Usage(embedding_tokens=count_tokens(prompt), completion_tokens=count_tokens(DONT_KNOW)),
                latency_ms=0.0,
            )

        # --- curation 1: entity-attribute consolidation (latest wins) ---
        latest_ts: dict[frozenset, float] = {}
        for _s, c in scored:
            ts = c.timestamp.timestamp()
            if c.topic and (c.topic not in latest_ts or ts > latest_ts[c.topic]):
                latest_ts[c.topic] = ts
        consolidated = [
            (s, c) for s, c in scored if not c.topic or c.timestamp.timestamp() >= latest_ts[c.topic]
        ]

        # --- curation 2: query-cue recency awareness ---
        # In 'recency' mode ("what's my current X?") the freshest *relevant* item
        # should win even if an older item is lexically more similar. We mark the
        # set that is relevant enough (cosine within RELEVANT_FRAC of the best)
        # and order that set by recency; in 'history'/'neutral' mode we keep the
        # standard cosine, recency-tiebreak order.
        if mode == "recency" and consolidated:
            max_cos = max(s for s, _c in consolidated)
            thresh = self.RELEVANT_FRAC * max_cos
            ranked = sorted(
                consolidated,
                key=lambda sc: (
                    0 if sc[0] >= thresh else 1,           # relevant items first
                    -sc[1].timestamp.timestamp(),          # then freshest
                    -sc[0],                                # then most similar
                    sc[1].item_id,
                ),
            )
        else:
            ranked = sorted(
                consolidated,
                key=lambda sc: (-sc[0], -sc[1].timestamp.timestamp(), sc[1].item_id),
            )

        top = ranked[:k]
        items = [
            MemoryItem(
                item_id=c.item_id,
                content=c.text,
                source_session=c.session_id,
                source_turn=c.turn_id,
                timestamp=c.timestamp,
                score=s,
                metadata={"consolidated": True},
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

    def consolidate(self) -> Usage:
        # Consolidation happens at query time from the immutable log; the hook is
        # a no-op here (kept for contract symmetry).
        return Usage()

    def stats(self) -> BackendStats:
        topics = {c.topic for c in self._chunks if c.topic}
        return BackendStats(
            item_count=len(self._chunks),
            bytes=sum(byte_size(c.text) for c in self._chunks),
            extra={"distinct_topics": len(topics)},
        )


__all__ = ["TemporalRAG", "topic_key"]

"""CuratedBrain adapter — the system-under-test, wired to the `curated-brain` package.

Curated Brain is a two-tier memory layer: a bi-temporal structured tier (entities +
`(subject, predicate, object)` facts with valid/transaction time and non-lossy supersede)
plus a vector tier, fed by a surprise-gated, self-organizing write path. The harness feeds
RAW turn text and never reveals gold (base.py rule 6), so this adapter drives CB with its
deterministic, no-LLM `HeuristicExtractor` — CB extracts its own triples from the text.

Scoring is provenance-based (`schema.resolve_item` maps a returned item's
`(source_session, source_turn)` to a gold fact id), so the adapter's central job is to
recover, for every memory CB surfaces, the originating turn. That needs NO change to CB's
core: we keep two ingest-time maps — by episodic record id (vector hits) and by the unique
per-turn timestamp CB stamps into every fact's provenance (structured facts, even when the
surprise gate discarded the raw episode).

Determinism: `HeuristicExtractor` + CB's `DeterministicEmbedder` + greedy curation are
byte-deterministic, so no `nondeterministic` flag is declared.
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
from mem_eval.pricing import price
from mem_eval.text import count_tokens

from curated_brain import CuratedBrain as _CuratedBrain
from curated_brain import HeuristicExtractor

# Each turn gets a unique, monotonically-increasing timestamp a hair above its session's
# instant — enough to identify the turn, far too small (ms) to cross a query's as_of (which
# the suite places days later), so session-level time-discipline is preserved.
_TURN_DELTA = 1e-3


class CuratedBrain(BaseBackend):
    name = "curated_brain"
    version = "0.1.0"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._cb = _CuratedBrain(extractor=HeuristicExtractor())
        self._turn_of_ts: dict[float, tuple[str, str]] = {}      # fact provenance wall_ts -> turn
        self._turn_of_episode: dict[str, tuple[str, str]] = {}   # episodic rid -> turn
        self._turn_meta: dict[tuple[str, str], tuple[str, datetime]] = {}  # turn -> (text, ts)

    def ingest(self, session: Session) -> Usage:
        base = session.timestamp.timestamp()
        emb_tokens = 0
        for i, turn in enumerate(session.turns):
            ts = base + i * _TURN_DELTA
            receipt = self._cb.write(turn.text, session_id=session.session_id, timestamp=ts)
            self._turn_of_ts[ts] = (session.session_id, turn.turn_id)
            if receipt.record_id is not None:  # stored (not gate-discarded) -> a vector record
                self._turn_of_episode[receipt.record_id] = (session.session_id, turn.turn_id)
            self._turn_meta[(session.session_id, turn.turn_id)] = (turn.text, session.timestamp)
            emb_tokens += count_tokens(turn.text)
        return Usage(embedding_tokens=emb_tokens, usd=price(embedding_tokens=emb_tokens))

    def consolidate(self) -> Usage:
        self._cb.consolidate()
        return Usage()

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        as_of_ts = as_of.timestamp()
        r = self._cb.query(prompt, session_id="__q__", timestamp=as_of_ts, k=k)
        stale = self._superseded_turns(as_of_ts)

        items: list[MemoryItem] = []
        seen: set[tuple[str, str]] = set()
        for cit in r.citations:
            origin = self._origin(cit)
            if origin is None or origin in seen or origin in stale:
                continue
            seen.add(origin)
            text, ts = self._turn_meta[origin]
            items.append(MemoryItem(
                item_id=f"{origin[0]}:{origin[1]}", content=text,
                source_session=origin[0], source_turn=origin[1], timestamp=ts))
            if len(items) >= k:
                break

        # Top-1 answer only (the resolved fact / best line), matching the baselines' single
        # `items[0].content` convention — fairer than handing the judge the whole context.
        answer = None
        if r.context:
            first = r.context.split("\n", 1)[0]
            answer = first.split("] ", 1)[1] if "] " in first else first
        emb = count_tokens(prompt)
        usage = Usage(embedding_tokens=emb, usd=price(embedding_tokens=emb))
        return QueryResult(items=items, answer=answer, usage=usage, latency_ms=0.0)

    def stats(self) -> BackendStats:
        st = self._cb.stats()
        return BackendStats(
            item_count=st.episodic_count + st.semantic_count + st.structured_count,
            bytes=len(self._cb.snapshot()),
            extra={"structured_facts": st.structured_count},
        )

    # --- provenance recovery -------------------------------------------------------------
    def _origin(self, citation) -> tuple[str, str] | None:
        """Map a CB citation back to its originating turn. Vector hits carry the episodic
        record id; structured facts carry their provenance ``wall_ts`` (set for every turn,
        even gate-discarded ones), so both paths resolve without touching CB's core."""
        o = self._turn_of_episode.get(citation.record_id)
        if o is not None:
            return o
        ts = citation.provenance.get("wall_ts")
        return self._turn_of_ts.get(ts) if ts is not None else None

    def _superseded_turns(self, as_of_ts: float) -> set[tuple[str, str]]:
        """Turns whose asserted fact was closed (superseded) by ``as_of`` — CB's own
        bi-temporal record, NOT gold. Used to drop stale items the token-based vector filter
        misses for multi-word values, so a superseded statement never counts as retrieved."""
        out: set[tuple[str, str]] = set()
        for f in self._cb.structured.facts:
            if f.valid_to <= as_of_ts:  # no longer valid at query time
                turn = self._turn_of_ts.get(f.provenance.get("wall_ts"))
                if turn is not None:
                    out.add(turn)
        return out


__all__ = ["CuratedBrain"]

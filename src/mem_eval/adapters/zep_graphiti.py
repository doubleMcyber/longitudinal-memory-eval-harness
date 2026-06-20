"""Zep adapter — via Graphiti, Zep's own open-source temporal-knowledge-graph engine.

Zep Community Edition needs a Docker server (Postgres + the Zep service), which isn't always
available. Graphiti (`graphiti-core`) is the *same engine* Zep is built on, and it runs fully
in-process against an **embedded Kuzu** graph DB (`db=":memory:"`) — no Docker, no server. The
LLM is driven through an OpenAI-compatible endpoint (e.g. a local Ollama/vLLM model, the SAME
model every system in the head-to-head uses), and the embedder + reranker are the harness's
deterministic ones so the embedding model is not a variable.

Provenance: Graphiti extracts entity-relationship *edges* (facts) from each episode; we record
`episode_uuid -> (session_id, turn_id)` at ingest, and at query time map each returned edge back
to a source turn via `edge.episodes`. That is exactly the `(source_session, source_turn)` the
harness scores against (PRD §4.2 rule 2).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

from mem_eval.adapters.base import (
    BackendStats,
    BaseBackend,
    MemoryItem,
    QueryResult,
    Session,
    Usage,
)
from mem_eval.text import count_tokens

from curated_brain.fakes import DeterministicEmbedder

_EMB_DIM = 256
_OPENAI_BASE = os.environ.get("ZEP_OPENAI_BASE", os.environ.get("MEM0_OPENAI_BASE", ""))
_OPENAI_MODEL = os.environ.get("ZEP_OPENAI_MODEL", os.environ.get("MEM0_OPENAI_MODEL", "qwen2.5:3b"))
_OPENAI_KEY = os.environ.get("ZEP_OPENAI_KEY", "ollama")


class _DetEmbedder:
    """Graphiti EmbedderClient backed by the harness deterministic embedder (fair: same embedder
    every backend uses). Graphiti calls ``create`` (single) and ``create_batch`` (list)."""

    def __init__(self) -> None:
        self.emb = DeterministicEmbedder(_EMB_DIM)

    async def create(self, input_data):
        text = input_data if isinstance(input_data, str) else " ".join(map(str, input_data))
        return [float(x) for x in self.emb.embed(text)]

    async def create_batch(self, input_data_list):
        return [[float(x) for x in self.emb.embed(t)] for t in input_data_list]


class _DetReranker:
    """CrossEncoderClient that ranks passages by cosine to the query under the deterministic
    embedder — keeps reranking LLM-free (no extra model calls), so cost stays comparable."""

    def __init__(self) -> None:
        self.emb = DeterministicEmbedder(_EMB_DIM)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []
        q = self.emb.embed(query)
        scored = [(p, float(self.emb.embed(p) @ q)) for p in passages]
        scored.sort(key=lambda ps: -ps[1])
        return scored


def _build_graphiti(model: str, base_url: str):
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    cfg = LLMConfig(api_key=_OPENAI_KEY, model=model, base_url=base_url, temperature=0.0)
    return Graphiti(
        graph_driver=KuzuDriver(db=":memory:"),
        llm_client=OpenAIGenericClient(config=cfg),
        embedder=_DetEmbedder(),
        cross_encoder=_DetReranker(),
    )


class ZepGraphiti(BaseBackend):
    name = "zep_graphiti"
    version = "0.1.0"

    def __init__(self) -> None:
        if not _OPENAI_BASE:
            raise RuntimeError(
                "ZepGraphiti needs an OpenAI-compatible endpoint: set ZEP_OPENAI_BASE "
                "(or MEM0_OPENAI_BASE), e.g. http://127.0.0.1:11434/v1 for local Ollama.")
        self._loop = asyncio.new_event_loop()
        self.reset()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def reset(self) -> None:
        self.g = _build_graphiti(_OPENAI_MODEL, _OPENAI_BASE)
        self._run(self.g.build_indices_and_constraints())
        self._ep_origin: dict[str, tuple[str, str]] = {}  # episode uuid -> (session, turn)
        self._turn_text: dict[tuple[str, str], tuple[str, datetime]] = {}
        self._usage = Usage()

    def ingest(self, session: Session) -> Usage:
        from graphiti_core.nodes import EpisodeType

        u = Usage()
        for i, turn in enumerate(session.turns):
            ts = session.timestamp
            self._turn_text[(session.session_id, turn.turn_id)] = (turn.text, ts)
            u.prompt_tokens += count_tokens(turn.text)
            res = self._run(self.g.add_episode(
                name=f"{session.session_id}:{turn.turn_id}",
                episode_body=turn.text,
                source=EpisodeType.text,
                source_description="conversation turn",
                reference_time=ts,
                group_id="eval",
            ))
            ep = getattr(res, "episode", None)
            if ep is not None:
                self._ep_origin[ep.uuid] = (session.session_id, turn.turn_id)
        self._usage += u
        return u

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        edges = self._run(self.g.search(prompt, num_results=k))
        items: list[MemoryItem] = []
        for e in edges:
            origin = None
            for ep_uuid in (e.episodes or []):
                if ep_uuid in self._ep_origin:
                    origin = self._ep_origin[ep_uuid]
                    break
            if origin is None:
                continue
            s, t = origin
            text, ts = self._turn_text.get((s, t), (e.fact, as_of))
            items.append(MemoryItem(
                item_id=f"{s}:{t}", content=e.fact, source_session=s, source_turn=t,
                timestamp=ts, score=None))
        answer = edges[0].fact if edges else None
        return QueryResult(items=items[:k], answer=answer, usage=Usage(), latency_ms=0.0)

    def stats(self) -> BackendStats:
        return BackendStats(item_count=len(self._ep_origin),
                            bytes=sum(len(t) for t, _ in self._turn_text.values()),
                            extra={"engine": "graphiti+kuzu", "model": _OPENAI_MODEL})


__all__ = ["ZepGraphiti"]

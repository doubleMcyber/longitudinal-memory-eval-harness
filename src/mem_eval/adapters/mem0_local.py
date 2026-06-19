"""Mem0 (the named rival) wired to run FULLY OFFLINE for a head-to-head.

Mem0 normally needs a cloud LLM + embedder. Here we drive it with a cached local model
(Qwen) behind mem0's `LLMBase`, the SAME deterministic embedder Curated Brain uses (so the
embedding model is not a variable), and an in-process qdrant store — no network, no keys.

Caveats (reported honestly, not hidden):
- Mem0 DISTILLS raw turns into rewritten fact strings, so provenance back to a source turn is
  lossy; we attach the source (session, turn) as metadata per add to give it the best shot at
  the harness's provenance-based recall/precision. The fair, provenance-independent metric for
  this comparison is **answer accuracy** (judge-graded), which mem0's design targets directly.
- The local model is small + CPU-bound, so this is a small-subset preliminary, not a full run.
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

from curated_brain.fakes import DeterministicEmbedder
from curated_brain.providers import TransformersLLM
from mem0.embeddings.base import EmbeddingBase
from mem0.llms.base import LLMBase

_EMB_DIM = 256
_MODEL = "Qwen/Qwen3.5-2B"
_SHARED_LLM = None  # load the local model once, reuse across scenario resets


def _shared_llm() -> TransformersLLM:
    global _SHARED_LLM
    if _SHARED_LLM is None:
        _SHARED_LLM = TransformersLLM(model_name=_MODEL, device="cpu", max_new_tokens=256)
    return _SHARED_LLM


class _Mem0LocalLLM(LLMBase):
    def __init__(self, config=None):
        self.llm = _shared_llm()

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        prompt = "\n\n".join(m.get("content", "") for m in messages)
        return self.llm.complete(prompt)


class _Mem0DetEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        self.emb = DeterministicEmbedder(_EMB_DIM)

    def embed(self, text, memory_action=None):
        return [float(x) for x in self.emb.embed(text)]


def _fresh_memory():
    import mem0.utils.factory as fac
    from mem0.vector_stores.qdrant import Qdrant
    from qdrant_client import QdrantClient
    fac.LlmFactory.create = staticmethod(lambda *a, **k: _Mem0LocalLLM())
    fac.EmbedderFactory.create = staticmethod(lambda *a, **k: _Mem0DetEmbedding())
    # Fresh IN-MEMORY qdrant per reset (no /tmp/qdrant disk lock across scenario resets).
    fac.VectorStoreFactory.create = staticmethod(lambda *a, **k: Qdrant(
        collection_name="h", embedding_model_dims=_EMB_DIM,
        client=QdrantClient(location=":memory:")))
    from mem0 import Memory
    return Memory.from_config({
        "vector_store": {"provider": "qdrant", "config": {"embedding_model_dims": _EMB_DIM}},
        "llm": {"provider": "openai", "config": {"model": "local"}},
        "embedder": {"provider": "openai", "config": {"model": "local"}},
    })


class Mem0Local(BaseBackend):
    name = "mem0_local"
    version = "0.1.0"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mem = _fresh_memory()
        self._uid = "harness"

    def ingest(self, session: Session) -> Usage:
        toks = 0
        for t in session.turns:
            self._mem.add(t.text, user_id=self._uid,
                          metadata={"session": session.session_id, "turn": t.turn_id})
            toks += count_tokens(t.text)
        # mem0 makes ~2 local-LLM calls per add (extract + update decision); price both.
        return Usage(prompt_tokens=toks * 2, usd=price(prompt_tokens=toks * 2))

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult:
        res = self._mem.search(prompt, filters={"user_id": self._uid}, top_k=k)
        rows = res.get("results", res) if isinstance(res, dict) else res
        items: list[MemoryItem] = []
        for r in rows:
            md = r.get("metadata") or {}
            items.append(MemoryItem(
                item_id=r.get("id", ""), content=r.get("memory", ""),
                source_session=md.get("session", ""), source_turn=md.get("turn"),
                timestamp=as_of, score=r.get("score")))
        answer = items[0].content if items else None
        emb = count_tokens(prompt)
        return QueryResult(items=items, answer=answer,
                           usage=Usage(embedding_tokens=emb, usd=price(embedding_tokens=emb)),
                           latency_ms=0.0)

    def stats(self) -> BackendStats:
        try:
            rows = self._mem.get_all(filters={"user_id": self._uid}).get("results", [])
        except Exception:
            rows = []
        return BackendStats(item_count=len(rows),
                            bytes=sum(len(r.get("memory", "")) for r in rows))


__all__ = ["Mem0Local"]

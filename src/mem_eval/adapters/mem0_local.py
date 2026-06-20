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

import os
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
from mem_eval.pricing import price
from mem_eval.text import count_tokens

from curated_brain.fakes import DeterministicEmbedder
from curated_brain.providers import OpenAICompatLLM, TransformersLLM
from mem0.embeddings.base import EmbeddingBase
from mem0.llms.base import LLMBase

_EMB_DIM = 256
# Local model + decode budget are overridable so a full-suite offline run can pick a smaller/
# faster cached model (e.g. Qwen/Qwen3-0.6B) without code edits. Defaults preserve prior runs.
_MODEL = os.environ.get("MEM0_MODEL", "Qwen/Qwen3.5-2B")
_MAX_NEW_TOKENS = int(os.environ.get("MEM0_MAX_NEW_TOKENS", "256"))
# When MEM0_OPENAI_BASE is set (e.g. a local Ollama/vLLM endpoint), drive mem0 through that
# OpenAI-compatible endpoint instead of CPU-transformers — the same fast shared model every
# system can use, which is what makes a full head-to-head tractable.
_OPENAI_BASE = os.environ.get("MEM0_OPENAI_BASE", "")
_OPENAI_MODEL = os.environ.get("MEM0_OPENAI_MODEL", "qwen2.5:3b")
_SHARED_LLM = None  # load the local model once, reuse across scenario resets


def _shared_llm():
    global _SHARED_LLM
    if _SHARED_LLM is None:
        if _OPENAI_BASE:
            _SHARED_LLM = OpenAICompatLLM(_OPENAI_MODEL, base_url=_OPENAI_BASE,
                                          api_key=os.environ.get("MEM0_OPENAI_KEY", "ollama"),
                                          max_tokens=_MAX_NEW_TOKENS)
        else:
            _SHARED_LLM = TransformersLLM(model_name=_MODEL, device="cpu",
                                          max_new_tokens=_MAX_NEW_TOKENS)
    return _SHARED_LLM


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Qwen3 family are reasoning models: their verbose <think> blocks are slow and break mem0's
# JSON/triple parsing. The "/no_think" soft switch disables reasoning; the regex strips any
# residual block. Off by default (preserves the documented n=3 reproduction); opt in with
# MEM0_NO_THINK=1. Measured caveat: even with it, a <=0.6B model is too weak to be a fair
# rival (answer_acc 0.0) AND mem0's many-calls-per-add makes a full run ~11h — see RESULTS.
_NO_THINK = os.environ.get("MEM0_NO_THINK", "0") == "1"

# Forced-JSON path: drive a FAST cached model (Qwen3-1.7B) with grammar-constrained decoding
# (outlines) so it ALWAYS emits valid JSON in mem0's EXACT expected shape. This rescues the fast
# model — which is plenty capable at extraction but emits prose, not JSON, by default — giving a
# fair SAME-MODEL CB-vs-mem0 comparison (the roadmap's literal Track-D). Opt in with MEM0_OUTLINES=1.
_OUTLINES = os.environ.get("MEM0_OUTLINES", "0") == "1"
_OUTLINES_MODEL = os.environ.get("MEM0_OUTLINES_MODEL", "Qwen/Qwen3-1.7B")
# mem0's two JSON calls (fact extraction → {"facts":[str]}, update → {"memory":[{id,text,event}]}).
_FACTS_SCHEMA = ('{"type":"object","properties":{"facts":{"type":"array","items":'
                 '{"type":"string"}}},"required":["facts"]}')
_MEM_SCHEMA = ('{"type":"object","properties":{"memory":{"type":"array","items":{"type":"object",'
               '"properties":{"id":{"type":"string"},"text":{"type":"string"},'
               '"event":{"type":"string"}},"required":["id","text","event"]}}},'
               '"required":["memory"]}')
_outlines_model = None
_outlines_gens: dict = {}


def _outlines_gen(schema: str):
    global _outlines_model
    import outlines
    if _outlines_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = AutoModelForCausalLM.from_pretrained(
            _OUTLINES_MODEL, dtype=torch.float16, attn_implementation="eager").to("mps")
        tok = AutoTokenizer.from_pretrained(_OUTLINES_MODEL)
        _outlines_model = outlines.from_transformers(hf, tok)
    if schema not in _outlines_gens:
        _outlines_gens[schema] = outlines.Generator(_outlines_model, outlines.json_schema(schema))
    return _outlines_gens[schema]


class _Mem0OutlinesLLM(LLMBase):
    """mem0 LLM driven by outlines forced-JSON over a fast cached model."""

    def __init__(self, config=None):
        pass

    def generate_response(self, messages, response_format=None, tools=None,
                          tool_choice="auto", **kwargs):
        prompt = "\n\n".join(m.get("content", "") for m in messages) + " /no_think"
        low = prompt.lower()
        # pick mem0's exact expected schema from the prompt (update vs extraction)
        if '"memory"' in low or "old_memory" in low or ('event' in low and 'update' in low):
            schema = _MEM_SCHEMA
        else:
            schema = _FACTS_SCHEMA
        try:
            return str(_outlines_gen(schema)(prompt, max_new_tokens=_MAX_NEW_TOKENS))
        except Exception:
            return '{"facts": []}'  # fail safe to valid-but-empty so mem0 doesn't crash


class _Mem0LocalLLM(LLMBase):
    def __init__(self, config=None):
        self.llm = _shared_llm()

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        prompt = "\n\n".join(m.get("content", "") for m in messages)
        if _NO_THINK:
            prompt += " /no_think"
        out = self.llm.complete(prompt)
        return _THINK_RE.sub("", out).strip() if _NO_THINK else out


class _Mem0DetEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        self.emb = DeterministicEmbedder(_EMB_DIM)

    def embed(self, text, memory_action=None):
        return [float(x) for x in self.emb.embed(text)]


def _fresh_memory():
    import mem0.utils.factory as fac
    from mem0.vector_stores.qdrant import Qdrant
    from qdrant_client import QdrantClient
    _llm_cls = _Mem0OutlinesLLM if _OUTLINES else _Mem0LocalLLM
    fac.LlmFactory.create = staticmethod(lambda *a, **k: _llm_cls())
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

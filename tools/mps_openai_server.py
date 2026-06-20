"""Minimal OpenAI-compatible endpoint backed by a local HF model on the Apple-Silicon GPU (MPS).

One shared, fast, offline model that every system in the head-to-head (Mem0 / Zep-Graphiti /
Letta / Curated Brain) can call through the standard `/v1/chat/completions` API — so the model
is held constant and the comparison is about *architecture*, not model. Embeddings are served
from the harness's deterministic embedder at `/v1/embeddings`, so the embedder is constant too.

Run:  PYTHONPATH=src MPS_MODEL=Qwen/Qwen3-1.7B python tools/mps_openai_server.py --port 11435
Then point clients at  http://127.0.0.1:11435/v1  (any api_key).

Deliberately tiny (stdlib http.server, single-threaded): the eval is sequential, and this keeps
the dependency surface at zero beyond what the harness already has.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# MPS generation is NOT thread-safe: two concurrent model.generate() calls on the same Metal
# model SIGSEGV. Mem0/graphiti issue several LLM calls per add, so serialize all generation.
_GEN_LOCK = threading.Lock()

from curated_brain.fakes import DeterministicEmbedder

_MODEL = os.environ.get("MPS_MODEL", "Qwen/Qwen3-1.7B")
_DEVICE = os.environ.get("MPS_DEVICE", "mps")
_MAX_NEW = int(os.environ.get("MPS_MAX_NEW_TOKENS", "512"))
_NO_THINK = os.environ.get("MPS_NO_THINK", "1") == "1"
_EMB_DIM = int(os.environ.get("MPS_EMB_DIM", "256"))
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_tokenizer = None
_model = None
_emb: DeterministicEmbedder | None = None


def _llm_once():
    """Load tokenizer+model directly in **fp16** on MPS. (TransformersLLM uses torch_dtype=auto
    -> bf16, which Metal runs pathologically slowly; fp16 is MPS's fast path.)"""
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # MPS needs fp16 (bf16 is <2 tok/s) AND eager attention (the fused SDPA path hits a
        # Metal grouped-query-attention matmul crash; fp16+SDPA also hangs on load). Eager is
        # slower (~4-5 tok/s) but is the only config that loads and runs without crashing here.
        dtype = torch.float16 if _DEVICE == "mps" else "auto"
        attn = "eager" if _DEVICE == "mps" else None
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL, dtype=dtype, attn_implementation=attn).to(_DEVICE)
    return _tokenizer, _model


def _emb_once() -> DeterministicEmbedder:
    global _emb
    if _emb is None:
        _emb = DeterministicEmbedder(_EMB_DIM)
    return _emb


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _generate(messages: list[dict]) -> str:
    """Native generation over the full message list with reasoning DISABLED for Qwen3-style
    models (enable_thinking=False in the chat template — the real switch; the soft '/no_think'
    token is unreliable through a plain prompt). Falls back gracefully if the template doesn't
    accept the kwarg. Reaches into the loaded TransformersLLM (._tok/._model/._dev)."""
    import torch

    tok, model = _llm_once()
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:  # template doesn't support enable_thinking (non-reasoning model)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(_DEVICE)
    with _GEN_LOCK, torch.no_grad():  # serialize: concurrent MPS generate() segfaults
        out = model.generate(**inputs, max_new_tokens=_MAX_NEW, do_sample=False, num_beams=1,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return _THINK_RE.sub("", tok.decode(gen, skip_special_tokens=True)).strip()


def _chat(body: dict) -> dict:
    messages = list(body.get("messages", []))
    # Nudge JSON when the client asked for it (mem0 / graphiti use response_format).
    rf = body.get("response_format") or {}
    if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        messages = messages + [{"role": "system", "content": "Respond with ONLY valid JSON."}]
    out = _generate(messages)
    pt = sum(_approx_tokens(m.get("content", "")) for m in messages)
    ct = _approx_tokens(out)
    return {
        "id": "chatcmpl-local", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", _MODEL),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": out},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


def _embeddings(body: dict) -> dict:
    inp = body.get("input", "")
    texts = inp if isinstance(inp, list) else [inp]
    emb = _emb_once()
    data = [{"object": "embedding", "index": i, "embedding": [float(x) for x in emb.embed(str(t))]}
            for i, t in enumerate(texts)]
    return {"object": "list", "data": data, "model": body.get("model", "det-embed"),
            "usage": {"prompt_tokens": sum(_approx_tokens(str(t)) for t in texts),
                      "total_tokens": sum(_approx_tokens(str(t)) for t in texts)}}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": _MODEL, "object": "model"}]})
        else:
            self._send(200, {"status": "ok", "model": _MODEL})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        try:
            if self.path.endswith("/embeddings"):
                self._send(200, _embeddings(body))
            else:
                self._send(200, _chat(body))
        except Exception as e:  # surface as a 500 the client can log
            self._send(500, {"error": {"message": repr(e)}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("MPS_PORT", "11435")))
    args = ap.parse_args()
    print(f"loading {_MODEL} on {_DEVICE} …", flush=True)
    _llm_once()
    print(f"ready: http://127.0.0.1:{args.port}/v1  (model={_MODEL})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()

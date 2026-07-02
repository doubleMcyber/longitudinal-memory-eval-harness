"""LongMemEval head-to-head: Curated Brain vs Mem0 vs Zep(Graphiti) [vs Letta] — Track D.

The named-rival run the roadmap has been blocked on, now executable locally: every system
drives the SAME local model (ollama, pinned tag) for its LLM calls, the SAME embedding model
(nomic-embed-text via ollama's OpenAI-compatible endpoint), the SAME answer-generation
prompt, and the SAME model-based judge (official LongMemEval yes/no style) — so the memory
layer is the only variable. Dataset: xiaowu0162/longmemeval (oracle or _s variant).

Usage:
  python3 bench_longmemeval.py --data /path/longmemeval_oracle --backends cb,mem0,zep \
      --n 500 --seed 42 --model qwen2.5:7b --out results/longmemeval/

Reproducibility: the question subset is a seeded stratified sample over question types;
model tags, prompts, and per-question raw outputs are all written to the output JSON.
Honest scope notes are printed into the results file (see `notes` key).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone

OLLAMA = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
K = 10  # retrieved memories per question, all systems

_ANSWER_PROMPT = (
    "You are a helpful assistant with access to memories from prior conversations with the "
    "user. Each memory line may include a date.\n\nMemories:\n{context}\n\n"
    "Current date: {qdate}\nQuestion: {question}\n\n"
    "Answer the question based ONLY on the memories above. Be concise. If the memories do "
    "not contain the answer, reply exactly: I don't know."
)
_JUDGE_PROMPT = (
    "You are grading a QA system.\nQuestion: {question}\nGold answer: {gold}\n"
    "System response: {resp}\n\nDoes the system response contain or entail the gold answer? "
    "Minor wording/format differences are fine; the meaning must match. Reply with exactly "
    "one word: yes or no."
)
_JUDGE_ABS_PROMPT = (
    "You are grading a QA system on an UNANSWERABLE question (the correct behavior is to "
    "abstain / say it does not know).\nQuestion: {question}\nSystem response: {resp}\n\n"
    "Did the system correctly abstain (say it doesn't know / lacks the information) instead "
    "of inventing an answer? Reply with exactly one word: yes or no."
)


# ------------------------------------------------------------------ shared model client --
class Counter:
    def __init__(self):
        self.calls = 0
        self.prompt_toks = 0
        self.completion_toks = 0


def _client():
    from openai import OpenAI
    return OpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama")


def chat(client, model: str, prompt: str, counter: Counter, max_tokens: int = 512) -> str:
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tokens)
    counter.calls += 1
    if r.usage:
        counter.prompt_toks += r.usage.prompt_tokens or 0
        counter.completion_toks += r.usage.completion_tokens or 0
    return (r.choices[0].message.content or "").strip()


def parse_date(s: str) -> datetime:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", s)
    y, mo, d, h, mi = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def session_text(session: list[dict], date: str) -> str:
    return "\n".join(f"[{date}] {t['role']}: {t['content']}" for t in session)


# ------------------------------------------------------------------------- backends ------
class CBBackend:
    """Curated Brain: shared-model session-batched extraction at write time + hybrid
    retrieval; the vector tier runs on the SAME nomic embedder as the rivals."""
    name = "curated_brain"

    def __init__(self, model: str, client, counter: Counter):
        from curated_brain.backend import CuratedBrain
        from curated_brain.providers import OpenAICompatEmbedder
        from curated_brain.surprise import SurpriseGate
        self.model, self.client, self.counter = model, client, counter
        emb = OpenAICompatEmbedder(EMBED_MODEL, dim=EMBED_DIM, base_url=f"{OLLAMA}/v1",
                                   api_key="ollama")
        # Real-embedder gate profile (disclosed): the defaults are calibrated for the
        # hashing test double, where unrelated texts score ~0 cosine. Real embeddings put
        # related chat turns at 0.6-0.8 cosine, so the default θ discards nearly all of a
        # session. Higher write budget + only near-verbatim reinforcement fits chat memory.
        gate = SurpriseGate(budget=0.8, theta0=0.25, theta_floor=0.10, reinforce_sim=0.92)
        self.cb = CuratedBrain(embedder=emb, dim=EMBED_DIM, gate=gate, max_context_items=K)

    _EXTRACT = (
        "Extract personal facts about the user from this conversation as lines of "
        "'subject | attribute | value'. Use 'User' as the subject for facts about the "
        "user. Only facts explicitly stated; if none, output NONE.\n\n{convo}\n\nFacts:")

    def ingest(self, session: list[dict], date: str) -> None:
        ts = parse_date(date).timestamp()
        # raw turns -> vector tier (episodic recall), surprise-gated
        for i, t in enumerate(session):
            self.cb.write(f"[{date}] {t['role']}: {t['content']}",
                          session_id=date, timestamp=ts + i)
        # ONE extraction call per session (write-time batching) -> structured tier
        user_text = "\n".join(t["content"] for t in session if t["role"] == "user")
        if not user_text.strip():
            return
        out = chat(self.client, self.model, self._EXTRACT.format(convo=user_text[:6000]),
                   self.counter, max_tokens=300)
        for line in out.splitlines():
            parts = [p.strip() for p in line.strip().lstrip("-*").split("|")]
            if len(parts) == 3 and all(parts) and parts[0].upper() != "NONE":
                subj, pred, obj = parts
                if pred.lower() == "attribute" or obj.lower() == "value":
                    continue  # the model echoed the prompt's template line, not a fact
                self.cb.write(f"[{date}] {subj}'s {pred} is {obj}.", session_id=date,
                              timestamp=ts + 1000,
                              metadata={"fact": {"subject": subj, "predicate": pred,
                                                 "object": obj}})

    def retrieve(self, question: str, qdate: str) -> str:
        from curated_brain.extraction import resolve_first_person
        self.cb.consolidate()  # CB's between-sessions design: dedupe fact copies, prune stale
        # the same first-person mechanism as the write path, applied to the question:
        # "Where did I ..." must reach the facts stored under the "User" subject
        q = resolve_first_person(question, "User")
        r = self.cb.query(q, session_id="q",
                          timestamp=parse_date(qdate).timestamp(), k=K)
        return r.context

    def stats(self) -> dict:
        m = self.cb.metrics()
        return {"items": m["store_size"], "facts": m["structured_facts"]}


class Mem0Backend:
    """Mem0 at its documented local configuration: ollama LLM + ollama embedder + qdrant."""
    name = "mem0"

    def __init__(self, model: str, client, counter: Counter):
        self.counter = counter
        # Fresh IN-MEMORY qdrant per question (the on-disk default holds a /tmp lock across
        # instances) — same approach as the proven mem0_local harness adapter.
        import mem0.utils.factory as fac
        from mem0.vector_stores.qdrant import Qdrant
        from qdrant_client import QdrantClient
        fac.VectorStoreFactory.create = staticmethod(lambda *a, **k: Qdrant(
            collection_name="h", embedding_model_dims=EMBED_DIM,
            client=QdrantClient(location=":memory:")))
        from mem0 import Memory
        self.mem = Memory.from_config({
            "llm": {"provider": "ollama",
                    "config": {"model": model, "temperature": 0.0,
                               "ollama_base_url": OLLAMA}},
            "embedder": {"provider": "ollama",
                         "config": {"model": EMBED_MODEL, "ollama_base_url": OLLAMA,
                                    "embedding_dims": EMBED_DIM}},
            "vector_store": {"provider": "qdrant",
                             "config": {"embedding_model_dims": EMBED_DIM}},
        })
        self.uid = "user"

    def ingest(self, session: list[dict], date: str) -> None:
        msgs = [{"role": t["role"], "content": t["content"]} for t in session]
        self.mem.add(msgs, user_id=self.uid, metadata={"date": date})
        self.counter.calls += 2  # mem0's extract + update calls (approximate accounting)

    def retrieve(self, question: str, qdate: str) -> str:
        res = self.mem.search(question, filters={"user_id": self.uid}, top_k=K)
        rows = res.get("results", res) if isinstance(res, dict) else res
        lines = []
        for r in rows[:K]:
            date = (r.get("metadata") or {}).get("date", "")
            lines.append(f"[{date}] {r.get('memory', '')}" if date else r.get("memory", ""))
        return "\n".join(lines)

    def stats(self) -> dict:
        try:
            rows = self.mem.get_all(filters={"user_id": self.uid}).get("results", [])
        except Exception:
            rows = []
        return {"items": len(rows)}


class ZepBackend:
    """Zep's engine (Graphiti) over embedded Kuzu — Docker-free, ollama LLM + embedder."""
    name = "zep_graphiti"

    def __init__(self, model: str, client, counter: Counter):
        self.counter = counter
        import asyncio
        self.aio = asyncio.new_event_loop()
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        llm = OpenAIGenericClient(config=LLMConfig(
            api_key="ollama", model=model, small_model=model, base_url=f"{OLLAMA}/v1"))
        emb = OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key="ollama", embedding_model=EMBED_MODEL, embedding_dim=EMBED_DIM,
            base_url=f"{OLLAMA}/v1"))
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        rer = OpenAIRerankerClient(config=LLMConfig(
            api_key="ollama", model=model, base_url=f"{OLLAMA}/v1"))
        driver = KuzuDriver(db=":memory:")
        # graphiti's (deprecated) Kuzu driver ships the FTS index DDL but never executes it
        # (build_indices_and_constraints is a no-op), so every fulltext search crashes with
        # "Table RelatesToNode_ doesn't have an index" — create them ourselves.
        import kuzu
        from graphiti_core.graph_queries import get_fulltext_indices
        from graphiti_core.driver.driver import GraphProvider
        conn = kuzu.Connection(driver.db)
        conn.execute("INSTALL FTS;")
        conn.execute("LOAD EXTENSION FTS;")
        for ddl in get_fulltext_indices(GraphProvider.KUZU):
            conn.execute(ddl)
        conn.close()
        self.g = Graphiti(graph_driver=driver, llm_client=llm, embedder=emb,
                          cross_encoder=rer)
        self.aio.run_until_complete(self.g.build_indices_and_constraints())

    def ingest(self, session: list[dict], date: str) -> None:
        from graphiti_core.nodes import EpisodeType
        self.aio.run_until_complete(self.g.add_episode(
            name=f"session-{date}", episode_body=session_text(session, date)[:8000],
            source=EpisodeType.message, source_description="chat session",
            reference_time=parse_date(date)))
        self.counter.calls += 4  # graphiti extract/dedupe/edge calls (approximate)

    def retrieve(self, question: str, qdate: str) -> str:
        results = self.aio.run_until_complete(self.g.search(question, num_results=K))
        lines = []
        for e in results[:K]:
            vf = getattr(e, "valid_at", None)
            lines.append(f"[{vf}] {e.fact}" if vf else str(e.fact))
        return "\n".join(lines)

    def stats(self) -> dict:
        return {}


BACKENDS = {"cb": CBBackend, "mem0": Mem0Backend, "zep": ZepBackend}


# ------------------------------------------------------------------------------ runner ---
def stratified_sample(data: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(data):
        return data
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    out = []
    per = max(1, n // len(by_type))
    for t in sorted(by_type):
        qs = sorted(by_type[t], key=lambda q: str(q["question_id"]))
        rng.shuffle(qs)
        out.extend(qs[:per])
    rng.shuffle(out)
    return out[:n]


def run_backend(kind: str, questions: list[dict], model: str, out_dir: str,
                tag: str) -> dict:
    client = _client()
    counter = Counter()
    judge_counter = Counter()
    rows = []
    t_start = time.time()
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            be = BACKENDS[kind](model, client, counter)
            for sess, date in zip(q["haystack_sessions"], q["haystack_dates"]):
                be.ingest(sess, date)
            context = be.retrieve(q["question"], q["question_date"])
            answer = chat(client, model, _ANSWER_PROMPT.format(
                context=context or "(no memories retrieved)", qdate=q["question_date"],
                question=q["question"]), counter)
            is_abs = str(q["question_id"]).endswith("_abs")
            jp = (_JUDGE_ABS_PROMPT.format(question=q["question"], resp=answer) if is_abs
                  else _JUDGE_PROMPT.format(question=q["question"], gold=q["answer"],
                                            resp=answer))
            verdict = chat(client, model, jp, judge_counter, max_tokens=5).lower()
            correct = verdict.startswith("yes")
            stats = be.stats()
            err = None
        except Exception as e:  # a backend crash on a question scores 0, run continues
            context, answer, correct, stats, err = "", "", False, {}, f"{type(e).__name__}: {e}"
        rows.append({"question_id": q["question_id"], "type": q["question_type"],
                     "correct": bool(correct), "answer": answer, "gold": q.get("answer"),
                     "context": context[:2000], "stats": stats, "error": err,
                     "seconds": round(time.time() - t0, 1)})
        done = sum(r["correct"] for r in rows)
        print(f"  [{kind}] {i + 1}/{len(questions)} acc={done / (i + 1):.3f} "
              f"({rows[-1]['seconds']}s){' ERR ' + err if err else ''}", flush=True)
        # checkpoint every question (long run; crash-safe)
        with open(os.path.join(out_dir, f"{kind}__{tag}.json"), "w") as fh:
            json.dump(_summarize(kind, model, rows, counter, judge_counter,
                                 time.time() - t_start), fh, indent=1)
    return _summarize(kind, model, rows, counter, judge_counter, time.time() - t_start)


def _summarize(kind, model, rows, counter, judge_counter, wall):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["correct"])
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else 0.0
    return {
        "backend": kind, "model": model, "n": n,
        "accuracy": round(acc, 4),
        "accuracy_by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
        "errors": sum(1 for r in rows if r["error"]),
        "cost": {"llm_calls": counter.calls, "prompt_tokens": counter.prompt_toks,
                 "completion_tokens": counter.completion_toks},
        "judge_cost": {"calls": judge_counter.calls},
        "wall_seconds": round(wall, 1),
        "notes": [
            "Same local model (ollama, pinned tag) for every system's LLM calls, answer "
            "generation, and judging; same nomic-embed-text embedder for every system.",
            "Judge is the shared local model with the official yes/no style prompt — NOT "
            "the official GPT-4o judge; treat absolute numbers as internally comparable, "
            "not as leaderboard-comparable.",
            "Mem0/Graphiti internal LLM calls counted approximately (2/add, 4/episode).",
        ],
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backends", default="cb,mem0,zep")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--out", default="results/longmemeval")
    args = ap.parse_args()

    data = json.load(open(args.data))
    questions = stratified_sample(data, args.n, args.seed)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{os.path.basename(args.data)}__n{len(questions)}__seed{args.seed}__" \
          f"{args.model.replace(':', '-').replace('/', '-')}"
    print(f"LongMemEval: {len(questions)} questions, model {args.model}, "
          f"backends {args.backends}")
    for kind in args.backends.split(","):
        print(f"== {kind} ==", flush=True)
        summary = run_backend(kind.strip(), questions, args.model, args.out, tag)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()

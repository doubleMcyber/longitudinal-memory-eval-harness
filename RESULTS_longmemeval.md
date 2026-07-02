# LongMemEval head-to-head — Curated Brain vs Mem0 vs Zep(Graphiti) vs Letta

First full named-rival run on externally-authored data (2026-07-02). Every system drove the
**same local model** (ollama `qwen2.5:7b`, temperature 0) for its LLM calls, answer
generation, and judging, and the **same embedder** (`nomic-embed-text` via ollama) — the
memory layer is the only variable. Runner: `bench_longmemeval.py` (this repo, pinned).

**Protocol** — dataset `xiaowu0162/longmemeval`, **oracle variant** (evidence sessions only,
mean 1.9 sessions / 21.9 turns per question); stratified seeded sample **n=138** (23 per
question type, seed 42) of the 500 questions; per question: fresh backend → ingest sessions
(with dates) → retrieve/answer → model-judged yes/no vs the gold answer (abstention variant
for `_abs` questions). Raw per-question outputs: `results/longmemeval/*.json`.

## Headline (accuracy = fraction judged correct, n=138)

| system | accuracy | wall time | LLM calls | notes |
|---|---|---|---|---|
| **Letta 0.16.8** | **0.471** | 159 min | ~730 (agentic) | 2 infra errors scored 0 |
| **Curated Brain** | 0.261 | **20 min** | **371** | 0 errors |
| **Mem0 2.0.7** | 0.203 | 77 min | ~601 | 1 error scored 0 |
| **Zep (Graphiti 0.29.2 + Kuzu)** | 0.065 | 254 min | ~1070 | 0 errors |

**Findings, stated plainly:**

1. **Curated Brain decisively beats Zep on accuracy (0.261 vs 0.065, McNemar p<0.0001) and
   beats both Mem0 and Zep on cost.** Its accuracy edge over Mem0 (0.261 vs 0.203) is
   **within noise at n=138 (paired McNemar p=0.24)** — a statistical tie on accuracy, a
   clear win on wall time (20 vs 77 min).
2. **Letta beats Curated Brain on accuracy (0.471 vs 0.261) at ~8× the wall time.** The
   roadmap's full claim ("≥ each of Mem0/Letta/Zep") is therefore **NOT met on this
   variant** — no spin: Letta wins the accuracy axis here.
3. **Why Letta wins this variant (and what would change at scale):** the oracle variant has
   ~2 evidence sessions per question, so Letta's agent effectively answers with the raw
   transcripts still in its context window — its memory machinery barely engages. It is
   operating as (agentic) full-context reading, the known accuracy ceiling that memory
   systems trade against. The **`longmemeval_s` variant** (~50-session, ~115k-token
   haystacks that overflow any context) is the setting where that mode breaks down and
   query-time retrieval quality decides — that run is the follow-up, not this document's
   claim.

## Per-type accuracy

| type (23 q each) | CB | Mem0 | Zep | Letta |
|---|---|---|---|---|
| knowledge-update | **0.478** | 0.391 | 0.087 | 0.435 |
| single-session-user | 0.522 | 0.435 | 0.087 | **0.565** |
| single-session-assistant | 0.261 | 0.043 | 0.130 | **0.957** |
| multi-session | 0.174 | 0.130 | 0.043 | **0.217** |
| single-session-preference | 0.087 | 0.130 | 0.000 | **0.217** |
| temporal-reasoning | 0.043 | 0.087 | 0.043 | **0.435** |

CB is strongest of all four on **knowledge-update** (belief revision — its bi-temporal
supersede design) and competitive on single-session-user; its clear general weaknesses are
**temporal-reasoning** (no date arithmetic over retrieved memories) and **preference**
questions (no preference-summary path) — logged as improvement areas, to be fixed generally
and re-run frozen, not tuned to this question set.

## Configuration disclosures (all in `bench_longmemeval.py`)

- **CB**: nomic embedder via `OpenAICompatEmbedder`; real-embedder gate profile
  (budget 0.8, θ₀ 0.25, floor 0.10, reinforce 0.92 — the defaults are calibrated for the
  hashing test double); session-batched LLM extraction (1 call/session) into the structured
  tier; first-person query rewrite ("I" → "User" — the same mechanism the write path uses);
  `consolidate()` before retrieval; k=10 context.
- **Mem0**: its documented local config (ollama LLM + ollama embedder), in-memory qdrant,
  `add(messages=whole-session)`; search top_k=10.
- **Zep**: Graphiti over embedded Kuzu (no Docker), one `add_episode` per session,
  `search(num_results=10)`. Note: graphiti's Kuzu driver is deprecated upstream and never
  creates its FTS indexes — we create them from graphiti's own DDL; Kuzu is not Zep's
  recommended production driver (Neo4j/FalkorDB are), which may understate Zep.
- **Letta**: its own server (embedded-postgres via `pgserver` + pgvector, schema from
  letta's ORM), agent per question, one message per session ("remember this conversation"),
  question answered agentically by Letta itself (its design); its token usage is not
  metered by our counter (calls approximate).

## Honest caveats

- **Judge**: the shared local `qwen2.5:7b` with a yes/no entailment prompt — NOT the
  official GPT-4o judge. Numbers are **internally comparable**, not leaderboard-comparable.
- **Oracle variant** ≠ the LongMemEval headline setting (`_s`); see finding 3.
- **n=138** stratified (seed 42): per-type cells are 23 questions; overall ±~0.08 at 95%.
- Letta's 2 errors and Mem0's 1 error (infra exceptions) are scored 0 for that question;
  error rows are marked in the raw JSONs.
- Session-batched ingest for every system (one add/episode/message per session) — a
  consistent framing, but not every system's most granular mode.
- Mem0/Zep/Letta internal LLM call counts are approximations (2/add, 4/episode, 2/turn).

## Reproduce

```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
python3 -c "import urllib.request,ssl,certifi;ctx=ssl.create_default_context(cafile=certifi.where());open('data/longmemeval_oracle','wb').write(urllib.request.urlopen('https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle',context=ctx).read())"
python3 bench_longmemeval.py --data data/longmemeval_oracle --backends cb,mem0,zep --n 140 --seed 42 --model qwen2.5:7b --out results/longmemeval
# letta additionally needs: pip install letta letta-client pgserver pgvector asyncpg ollama
# + an embedded-postgres holder and `letta server` (see bench_longmemeval.py LettaBackend)
python3 bench_longmemeval.py --data data/longmemeval_oracle --backends letta --n 140 --seed 42 --model qwen2.5:7b --out results/longmemeval
```

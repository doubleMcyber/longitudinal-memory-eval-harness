# LongMemEval head-to-head — Curated Brain vs Mem0 vs Zep(Graphiti) vs Letta

First full named-rival run on externally-authored data (2026-07-02/03). Every system drove the
**same local model** (ollama `qwen2.5:7b`, temperature 0) for its LLM calls, answer
generation, and judging, and the **same embedder** (`nomic-embed-text` via ollama) — the
memory layer is the only variable. Runner: `bench_longmemeval.py` (this repo, pinned).

**The one-paragraph result.** The outcome is **regime-split**, and neither regime is spin:
on the **oracle** variant (~2 evidence sessions/question — the history *fits* the model's
context) **Letta wins** (0.471 vs CB 0.261) because its agent effectively reads the raw
transcripts and its memory machinery barely engages. On the **`_s`** variant (~50 sessions,
~490k chars/question — the history *overflows* any context) that mode collapses: **CB ties
the best system (0.167, tied with Mem0) and beats Letta (0.083, partial), at 8–24× lower
cost per question**, and Zep fails to complete a single question. So the roadmap's
unconditional "CB ≥ each of Mem0/Letta/Zep" is **not** met (Letta wins oracle); but there is
a real regime — long histories that don't fit context, i.e. the problem a memory layer
*exists* to solve — where CB is the accuracy co-leader and the clear cost leader.

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

## The `_s` variant — long histories that overflow context (2026-07-03)

The oracle result above is the *easy* regime for an agentic reader. The `longmemeval_s`
variant is the setting a memory layer actually targets: **mean 50 sessions / 494 turns /
~490k characters per question** — far beyond the model's 32k-token window, so no system can
just read the transcript. Same protocol, same shared model/embedder/judge; stratified
**n=24** (4 per type, seed 42). Because several rivals are extremely slow at this scale, two
legs are **disclosed partials** (pre-registered cut rules), not full runs:

| system | accuracy | n | cost/question | status |
|---|---|---|---|---|
| **Curated Brain** | **0.167** | 24 | **3.1 min** | complete, 0 errors |
| **Mem0 2.0.7** | **0.167** | 24 | 25.1 min | complete, 0 errors |
| Letta 0.16.8 | 0.083 | 12 | 70.4 min | **partial** (cut at n=12 per rule) |
| Zep (Graphiti+Kuzu) | — | 0 | >120 min | **DNF** (see below) |

**Findings, stated plainly:**

1. **CB and Mem0 tie exactly at 0.167** (both 24/24; paired: 3 CB-only-correct, 3
   Mem0-only-correct, 1 both — a dead heat), but **CB is 8× cheaper** (3.1 vs 25.1 min/q;
   1153 vs 2434 LLM calls). At the scale a memory layer is *for*, CB matches the strongest
   rival's accuracy at a fraction of the cost.
2. **Letta collapses from 0.471 (oracle) to 0.083 (`_s`)** — below CB — and is the slowest
   viable system (70 min/q, ~23× CB). This is the predicted result: once history overflows
   context, its agentic full-context advantage is gone and it must rely on memory recall,
   where it underperforms. (Cut at n=12 after a pre-registered rule: ≤2/12 correct at
   ~70 min/q → the ~24 h remaining runtime was not worth it; 1/12 correct. Directional, wide
   CI.)
3. **Zep did not complete a single `_s` question in ~2 hours** and was cut. Graphiti issues
   ~200 LLM calls per question to build its graph over 50 sessions; on a local 7B that is
   throughput-infeasible. This is a real finding about graph-construction cost at scale, not
   an omission — but it means we have **no `_s` accuracy number for Zep** (it was last at
   oracle, 0.065).
4. **Everyone's absolute accuracy is low at `_s`** (0.083–0.167) — 490k-char haystacks with
   a 7B model and a token-cosine-free retrieval budget of k=10 is genuinely hard, and the
   local 7B judge is strict. These are directional at n=24 (±~0.15 at 95%); the *ordering*
   (CB = Mem0 > Letta > Zep-DNF) is the result, not the absolute values.

**`_s` disclosures (beyond the shared ones below):** n=24 is small — treat as directional.
Three runs failed on infrastructure and were re-run or cut, archived as `.bak` for audit:
`letta__FAILED_TIMEOUTS` (first `_s` letta attempt — every question hit the SDK's default
HTTP timeout; fixed with `timeout=1800`), `mem0__FAILED_EMPTYEMBED` (mem0's ollama embedder
raised on empty extracted memories; patched to substitute a space — disclosed in code). Episode
truncation was raised 8k→24k chars for `_s` (at ~10k chars/session the 8k cap silently cut
~20% of each session). CB's config is unchanged from the oracle run (frozen).

## Overall verdict on the DONE clause

The roadmap's bar was "LongMemEval shows CB ≥ each of Mem0/Letta/Zep on the headline metric
at ≤ its cost." Measured, unspun:

- **vs Mem0:** CB ties on accuracy on *both* variants (0.261~0.203 oracle within noise;
  0.167=0.167 `_s` exact) and wins decisively on cost on both → **CB ≥ Mem0 holds.**
- **vs Zep:** CB beats it on oracle accuracy (0.261 vs 0.065) and cost, and Zep is
  infeasible at `_s` → **CB ≥ Zep holds.**
- **vs Letta:** split — **Letta wins oracle (0.471 vs 0.261); CB wins `_s` (0.167 vs
  0.083).** So **CB ≥ Letta does NOT hold unconditionally.** The unconditional DONE clause
  is **not met.**

The honest headline: **Curated Brain is the accuracy co-leader and the runaway cost leader
in the regime a memory layer is built for (histories that don't fit context), and it beats
or ties Mem0 and Zep everywhere — but Letta beats it when the whole history fits the model's
window.** That is a real, defensible position; it is not the unconditional win the clause
demands.

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

# the _s variant (long histories): same command, --data data/longmemeval_s --n 24 --out results/longmemeval_s
# (fetch longmemeval_s the same way; note zep is throughput-infeasible at _s on a local 7B)
```

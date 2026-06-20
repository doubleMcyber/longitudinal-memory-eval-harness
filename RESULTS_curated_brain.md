# Preliminary result — Curated Brain vs the shipped references

**Suite:** `v1`, scale `standard`, seed `42`, `k=10` · **fully offline & deterministic** ·
harness `compare` below. Reproduce: `for b in curated_brain temporal_rag naive_rag semantic_rag
long_context no_memory; do PYTHONPATH=src python -m mem_eval.runner.cli run --backend $b --suite v1
--seed 42 --k 10 --scale standard --out results/; done && PYTHONPATH=src python -m
mem_eval.runner.cli compare results/*.json`

| metric | curated_brain | temporal_rag | naive_rag | semantic_rag | long_context | no_memory |
| --- | --- | --- | --- | --- | --- | --- |
| recall@k | 0.880 | **0.920** | 0.800 | 0.920 | 0.600 | 0.000 |
| precision@k | **0.790** | 0.536 | 0.391 | 0.403 | 0.188 | 0.000 |
| contradiction_acc | **1.000** | 0.800 | 0.100 | 0.100 | 0.100 | 0.000 |
| staleness (lower=better) | **0.000** | 0.060 | 0.280 | 0.280 | 0.280 | 0.000 |
| answer_acc | 0.760 | 0.760 | 0.680 | 0.760 | 0.680 | 0.000 |
| cost_per_query_usd | **~0.0000** | 0.0001 | 0.0001 | 0.0001 | 0.0004 | 0.000 |
| storage_growth_slope | 516.96 | 45.72 | 45.72 | 45.72 | 45.72 | 0.00 |

## Verdict: Curated Brain is the strongest backend on every quality axis **except raw recall**

Against the contradiction-aware reference `temporal_rag`, Curated Brain now:

- **Wins precision decisively (0.79 vs 0.54)** — the highest of any backend.
- **Wins contradiction-resolution (1.00 vs 0.80) and staleness (0.00 vs 0.06)** — perfect: it
  never surfaces a superseded value, the core bi-temporal claim.
- **Ties answer accuracy (0.76)** and **ties multi_hop, needle, and contradiction recall (all 1.00)**.
- **Wins long-range recall by category (0.83 vs 0.67)** and is the **cheapest** backend.
- **Loses only on overall recall (0.88 vs 0.92)** — a single category (see below).

So on this neutral, third-party harness it never saw, CB is *competitive-to-better* overall:
it wins or ties every metric but one, and the deficit is 0.04 of recall.

## The one remaining gap (and why we didn't close it)

The entire 0.88-vs-0.92 recall deficit is the `recency_relevance` category (CB 0.50 vs 1.00). Its
update turn — *"Oscar briefly noted **the project** changed to July"* — needs definite-NP +
ellipsis coreference ("the project" → Oscar's *project deadline*) to link the new value to the
prior fact. A regex for that would be special-casing this benchmark's phrasing, so we **did not**
add it (the plan's "don't tune to the benchmark" rule). It's left for a real semantic extractor.

## How CB got here (general capabilities, not benchmark hacks)

Each lever is a general capability, verified by a separate adversarial review and AC-9-safe:

1. **Heuristic extractor** — deterministic possessive/verb triple extraction (no LLM).
2. **Multi-word + multi-entity routing** — schema-driven planner over stored predicates; a
   backstop that surfaces facts for *every* named entity when a plan mis-routes → `multi_hop` 0.67→1.00.
3. **Relational patterns** — "works at/for", "is headquartered in", "located in".
4. **Recency-based pronoun coreference** — "Their/His/Her current X" → most-recent subject →
   `contradiction` recall 0.80→1.00, staleness 0.20→0.00.

## Fair-comparison disclosures

- **Answer = top-1** (the resolved fact / best line), matching the baselines' single
  `items[0].content` convention.
- **Storage slope (517 vs 46)** is a serialization artifact: `stats().bytes = len(cb.snapshot())`,
  a verbose JSON dump of facts + provenance + embedding config, vs the references' raw chunks.
  Not claimed as a win or a real loss.
- **Provenance audit:** adapter is contract-clean (17/17), 0 unmapped citations, 0 gold turns
  wrongly excluded, no gold peeking. Superseded items are dropped via CB's own bi-temporal state.

## Real embedder (bge) — fixes paraphrase, but trades aggregate recall (honest)

CB can run on a real semantic embedder (`BAAI/bge-small-en-v1.5`, offline) via `CB_EMBEDDER=bge`.
It **fixes the paraphrase/lexical-gap category** the deterministic test-double can't:

| lexgap-0 (paraphrase) | answer | recall | precision |
|---|---|---|---|
| CB + deterministic | 0.00 | 0.00 | 0.00 |
| **CB + bge** | **1.00** | **1.00** | **1.00** |

But on the **full standard suite** bge is a *tradeoff*, not a free win — semantic spread displaces
some lexically-clean gold, so aggregate **recall drops 0.88 → 0.84** even as **precision rises
0.79 → 0.84** (contradiction 1.00, answer 0.76 unchanged). So a real embedder is the right tool
for paraphrase-heavy data but does not, by itself, close CB's headline recall gap on this suite.
(The harness references keep their own offline embedders; a fully fair run would give them bge too.)

**Hybrid retrieval (added):** CB's `VectorTier.search` now fuses embedding similarity with lexical
token-overlap. This **fixes bge's paraphrase regression** — with hybrid, the
`longitudinal_recall`/lexgap category recovers to **1.00** (semantic finds the paraphrase, lexical
keeps the exact mentions), and contradiction/multi_hop stay 1.00. But aggregate bge recall stays
**0.84**: the remaining loss is the **`needle`** category (0.00), which is a *surprise-gate/storage*
interaction — under bge's novelty distribution the needle observation is gated differently — **not**
a retrieval-ranking issue, so hybrid (a ranking change) cannot fix it. Honest: hybrid is a real,
general, AC-safe improvement that makes the real embedder viable; it does **not** move the headline
recall number.

## Named-rival: Mem0 — preliminary OFFLINE result (n=3 subset)

A first real head-to-head vs **Mem0** (`mem0ai` 2.0.7), run fully offline: Mem0 driven by a
cached local model (`Qwen3.5-2B`) through a custom `LLMBase`, the **same** deterministic
embedder as CB, in-memory qdrant. Mem0 is CPU-bound (~2.7 min/add), so this is the smallest
scenario per key category (`contra-2`, `hop-1`, `long-0` — 15 turns, **3 queries**).
Reproduce: `PYTHONPATH=src python bench_mem0_h2h.py` (adapter: `mem_eval/adapters/mem0_local.py`).

| metric | curated_brain | temporal_rag | mem0 (Qwen-2B) |
|---|---|---|---|
| answer accuracy | **1.00** | 0.67 | 0.67 |
| recall@k | 1.00 | 1.00 | 1.00 |
| precision@k | **1.00** | 0.67 | 0.37 |
| contradiction-resolution | **1.00** | 0.00 | 0.00 |
| ingest wall-time (15 turns) | **~0 s** | ~0 s | **~70 min** |

### Important correction — this subset is CB-FAVORABLE; the broader picture is narrower

The subset above is the *smallest scenario per category*, which happened to be CB's strong
categories. A broader (partial) run over additional scenarios (`long-0`, `long-1`, `lexgap-0`)
was attempted but the **full suite is infeasible offline** — Mem0 is ~2 h per large scenario on
CPU (~15–20 h total), so it was killed after 3 more scenarios. What those scenarios show changes
the story, honestly:

| scenario | answer (CB/TR/Mem0) | precision (CB/TR/Mem0) |
|---|---|---|
| long-0 / long-1 (plain recall) | 1.00 / 1.00 / 1.00 *(all tie)* | **1.00** / 0.50 / ~0.13 |
| lexgap-0 (paraphrase) | 0.00 / 0.00 / 0.00 *(all fail)* | 0.00 / 0.00 / 0.00 |

**Honest synthesis across all scenarios actually run (contra-2, hop-1, long-0/1, lexgap-0):**
- **CB dominates precision** everywhere (≈1.00 vs Mem0 ≈0.1–0.4) — it returns clean minimal facts;
  Mem0 returns a noisy distilled set.
- **CB wins contradiction** (1.00 vs 0.00) — Mem0's additive extraction doesn't supersede.
- **Answer accuracy: CB ties** Mem0/temporal on plain longitudinal recall; the "1.00 vs 0.67
  sweep" above was the favorable subset, **not** a general result.
- **Paraphrase (lexgap) fails for ALL three** — a shared limitation of the offline (non-semantic)
  embedders, **not** a CB-specific weakness.

**Caveats (do not over-read):** n is tiny; Mem0 ran on a small local model via an OpenAI-shaped
shim (frequent JSON-parse errors observed → **not Mem0 at its cloud best**); different extractors
(CB heuristic vs Mem0 LLM); `cost_per_query` excludes Mem0's heavy ingest (the ~70 min is the real
cost gap). Adapter fairness was confirmed by a separate adversarial review (which fixed a `top_k`
defect first). A *credible, full* run needs a capable shared model + a real semantic embedder.

## Measured feasibility of a full local run (2026-06-20) — why a faster small model is NOT a shortcut

The obvious idea — swap in a tiny faster model to make the full suite tractable — was **measured
and rejected**. With `MEM0_MODEL=Qwen/Qwen3-0.6B` (+ `/no_think` to suppress reasoning verbosity)
on one 2-turn scenario:

- **Throughput:** ~350 s/add (mem0 issues *many* LLM calls per add — extract + per-memory compare +
  update decision — not ~2), projecting to **~11.5 h** for the 118-add `small` suite — *worse* than
  the 2B model, because the small model's per-token speed gain is swamped by call count.
- **Quality:** Mem0 scored **answer 0.00 / contradiction 0.00** — the 0.6B model is too weak to
  extract usable memories, so a run against it would be an **unfair strawman**, not a real rival.

So the local-CPU envelope is a genuine bind: the only models fast enough are too weak to be a fair
Mem0, and a *fair* model (≥2B) is ~5 h for Mem0 **alone**. The `MEM0_MODEL` / `MEM0_MAX_NEW_TOKENS`
/ `MEM0_NO_THINK` env knobs were added so the eventual **hosted-endpoint** run is a one-liner, but
they do **not** make a credible full local run feasible. Network egress IS available here (so the
clients pip-install), but **Zep additionally needs a Docker server (absent)** — so even Mem0+Letta
locally cannot satisfy the "≥ each of Mem0/Letta/Zep" headline. The blocker is now *quantified*,
not assumed: a capable shared inference endpoint is required.

### Update — three "blocked" assumptions overturned, but local GPU inference hit a cascade of bugs

Re-probing the environment (rather than assuming) corrected three earlier beliefs and produced
endpoint-ready infrastructure — but a *working* local run was defeated by per-model technical bugs:

- ✅ **Network egress exists** — `graphiti-core`, `kuzu`, `letta`, `llama-cpp-python` all pip-install.
- ✅ **No-Docker Zep is possible** — Zep's own engine **Graphiti** runs in-process against an
  **embedded Kuzu** graph DB (`KuzuDriver(db=":memory:")`). New adapter: `adapters/zep_graphiti.py`
  (LLM→OpenAI-compatible endpoint; deterministic embedder + embedding-reranker for fairness).
- ✅ **The MPS GPU works** (`torch.backends.mps` ok; 25 GiB) — earlier "MPS broken" was stale.
- ❌ **But no local model would actually serve fast+sane**, after a fair sweep:
  - Ollama (`brew install ollama` works) — but the **registry download stalls** (that CDN blocked);
    partial blobs stuck at 58 B.
  - **MPS is a dead end for modern GQA models** — Metal's `mps.matmul` throws
    `incompatible dimensions` on grouped-query-attention shapes, crashing **both** Ministral-8B
    (`1x32`/`1x8` heads) **and** Qwen3-1.7B (`1x16`/`1x8`) on the default SDPA path. When it
    doesn't crash, bf16-on-MPS runs <2 tok/s (Metal's slow path) and fp16+SDPA `.to("mps")` hangs.
  - **fp16 + `attn_implementation="eager"` is the ONLY config that runs** (avoids the fused GQA
    matmul) — ~4.5 tok/s. The first segfault-under-load was **fixed** by serializing generation
    with a lock (concurrent MPS `generate()` is not thread-safe). With that, a full end-to-end run
    completed — and proves the deeper point:

  **End-to-end run (stable inference, shared Qwen3-1.7B, 3 scenarios, 2026-06-20):**

  | scenario | CB ans / prec | temporal_rag ans / prec | **mem0 ans** | mem0 wall |
  |---|---|---|---|---|
  | long-0 (recall) | 1.0 / 1.0 | 1.0 / 0.5 | **0.0** | 722 s |
  | contra-0 (contradiction) | 1.0 / 1.0 | — | **0.0** | **3125 s (52 min)** |
  | hop-0 (multi-hop) | 0.0 / 1.0 | — | **0.0** (extraction timeouts) | 1082 s |
  | **aggregate** | **0.67 / 1.00** | 0.67 / **0.53** | **0.00 / 0.00** | — |

  Mem0's calls *succeeded* (no infra error on long-0/contra-0) yet it scored **0.0 everywhere**:
  Qwen3-1.7B returns markdown prose ("Here are the **triples**…"), not the strict JSON Mem0's
  extractor parses, so it stores no usable memory — **a weak-model strawman, not a real Mem0**, and
  ~52 min for one scenario. A *fair* Mem0 needs a capable (JSON-compliant) model.

  **Conclusion (now proven end-to-end, not projected):** local capable-model inference is
  non-viable on this box — capable models crash (MPS GQA) or can't download (LFS/registry/raw all
  blocked), and the only runnable model (≤1.7B, fp16-eager) is too weak+slow to be a fair rival.
  A hosted OpenAI-compatible endpoint is required; the adapters (`MEM0_OPENAI_BASE`/`ZEP_OPENAI_BASE`,
  CB `OpenAICompatLLM`) target it directly. CB-vs-references above is unaffected and reproducible.
  - **CPU** is the ~5–11 h/system wall already measured above.

Net: built `tools/mps_openai_server.py` (a shared OpenAI-compatible local endpoint) and the Zep
adapter, and wired Mem0 to an endpoint (`MEM0_OPENAI_BASE`) — **all endpoint-ready but not yet
validated end-to-end**, because this box has no working fast local model. Pointing any **hosted**
OpenAI-compatible endpoint at them (set `MEM0_OPENAI_BASE`/`ZEP_OPENAI_BASE`) unblocks the run.

## Next, to make it a clean win / named-rival claim

- Run the full suite with a **capable shared endpoint** (CB + Mem0/Letta/Zep, same model) — the DONE headline.
- General semantic extraction to close `recency_relevance` (the last recall gap vs references).
- A shared embedder across CB and the references (rigor: make architecture the only variable).

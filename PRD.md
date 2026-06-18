# PRD — Longitudinal Memory Eval Harness

**Status:** v1 spec · **Date:** 2026-06-17 · **Type:** implementation-ready engineering spec
**Gate:** `pytest -q` exits 0 against the acceptance suite defined in §11.

---

## TL;DR

There is currently no reliable way to tell whether an agent's memory system actually works *over time*. In-context "needle in a haystack" tests measure retrieval inside a single context window; they say nothing about persistence across sessions, contradiction handling, or curation quality. This project builds a **benchmark + runner** that measures how well a memory system **retains and uses information across sessions**.

The spine of the system is a **backend adapter interface** — a single, frozen contract that any memory system implements (no-memory, long-context stuffing, naive-RAG, third-party systems like Letta, and the Curated Brain). Plug in a backend and get a **reproducible scorecard**: recall@k, precision, contradiction-resolution accuracy, staleness, query latency, cost per query, and storage growth over time.

The harness ships with **baseline backends in-box** so every score is relative to a reference point, and with **synthetic data generation that carries exact ground-truth labels** (plus a defined path to run on real conversation logs). Success is when an improvement or regression in any memory system shows up immediately and unambiguously as a change in these numbers. This is the permanent scoreboard everything else is judged against.

The single sharpest acceptance test (§11, A4): the harness must **discriminate** — on the v1 suite, no-memory must score strictly worse than naive-RAG on recall, and naive-RAG must exhibit measurable staleness on the contradiction suite. If the harness can't separate known-different baselines, it is broken, regardless of how clean the code is.

---

## 1. Problem & motivation

Agent memory is currently unfalsifiable. Teams ship "memory" features and judge them by vibes or by single-session retrieval tests that don't capture the actual failure modes that matter in long-lived agents:

- **Persistence** — a fact stated 40 sessions ago should still be usable today.
- **Contradiction handling** — when a fact is updated, the *new* value must win; the stale value must not resurface.
- **Curation quality** — memory that grows without bound, or that returns outdated/irrelevant items, degrades silently.

Because these are cross-session properties, they are invisible to in-context benchmarks. The result: nobody can say whether memory system A is better than B, or whether last week's change was an improvement or a regression. This harness makes memory quality **falsifiable and comparable**.

## 2. Goals / non-goals

**Goals (v1)**

- Define one stable **adapter interface** that is the central contract for every backend.
- Ship **3 baseline backends in-box** (no-memory, long-context, naive-RAG) so all scores are relative.
- Cover **5 test categories** and emit **7 metrics** in a machine-readable, reproducible scorecard.
- Generate **synthetic multi-session data with exact ground-truth labels**, deterministically by seed.
- Define and partially implement a **path to real conversation logs**.
- Make runs **reproducible** (same seed + backend version + config ⇒ identical metric hash).

**Non-goals (v1)**

- Not building the Curated Brain itself — it is one backend the harness *judges* (shipped as a stub adapter).
- Not optimizing any backend for score; the harness measures, it does not tune.
- Not a production serving system, a UI, or a hosted leaderboard (a local `compare` view is in scope; a hosted board is roadmap).
- Not a single-context-window / in-prompt retrieval benchmark — that is the thing this explicitly replaces.

## 3. Core concept & terminology

- **Session** — one bounded conversation occurring at a timestamp. Memory is written *between* sessions and read *within* later sessions.
- **Turn** — a single message inside a session (the unit a fact can be planted on).
- **Fact** — a structured `(entity, attribute, value, valid_from)` assertion planted in a session; the atomic unit of ground truth.
- **Update / contradiction** — a later fact that supersedes an earlier one for the same `(entity, attribute)`. The earlier version becomes **superseded** as of the update.
- **Scenario** — an ordered timeline of sessions plus the planted gold structure and the queries derived from it.
- **Query** — a probe issued at an `as_of` time, with a gold answer and a gold **supporting-item set** (the fact IDs that should be retrieved), tagged with a test category.
- **Backend** — a memory system implementing the adapter interface.
- **Scorecard** — the JSON + human-readable output of a `(backend × suite)` run.

## 4. The backend adapter interface — the central contract

This is the most important section. Everything else depends on it, and it is **frozen first** (Stage 1). Adding a backend means implementing this and nothing else; the harness never special-cases a backend.

### 4.1 Data types

```python
@dataclass(frozen=True)
class Turn:
    turn_id: str
    role: str                 # "user" | "assistant" | "system"
    text: str

@dataclass(frozen=True)
class Session:
    session_id: str
    timestamp: datetime       # when this session occurred
    turns: list[Turn]

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    usd: float = 0.0

@dataclass
class MemoryItem:
    item_id: str              # stable id the harness can grade against gold support
    content: str
    source_session: str
    source_turn: str | None
    timestamp: datetime       # provenance: when the underlying info was first seen
    score: float | None = None  # backend's own relevance score (optional)
    metadata: dict = field(default_factory=dict)

@dataclass
class QueryResult:
    items: list[MemoryItem]   # ranked best-first; graded against gold support
    answer: str | None        # optional synthesized answer; graded when present
    usage: Usage
    latency_ms: float

@dataclass
class BackendStats:
    item_count: int
    bytes: int                # storage footprint; basis for storage-growth metric
    extra: dict = field(default_factory=dict)
```

### 4.2 The interface

```python
class MemoryBackend(Protocol):
    name: str
    version: str              # bumped on any behavior change; recorded in every scorecard

    def reset(self) -> None: ...
        # wipe all state; harness calls this before each scenario for isolation.

    def ingest(self, session: Session) -> Usage: ...
        # persist one session into memory. Called in timestamp order.

    def query(self, prompt: str, k: int, as_of: datetime) -> QueryResult: ...
        # retrieve up to k items relevant to prompt, usable for answering.

    def consolidate(self) -> Usage: ...
        # optional background curation/compaction hook; default no-op.
        # called by the harness at configured cadences to test curation systems.

    def stats(self) -> BackendStats: ...
        # current storage footprint and item count.
```

### 4.3 Contract rules (non-negotiable, enforced by tests)

1. **Time discipline.** `query(as_of=T)` MUST only use information from sessions with `timestamp <= T`. This is what makes "longitudinal" real and contradiction tests meaningful. The harness verifies this with a leakage probe (query before a fact is ingested must not return it).
2. **Provenance.** Every returned `MemoryItem` MUST carry `source_session` and `timestamp` so the harness can grade against gold supporting-item IDs and compute staleness. `item_id` must be stable within a run.
3. **Isolation.** `reset()` fully clears state; no scenario may leak into another.
4. **Determinism (or declared nondeterminism).** Given the same ingest order, config, and `as_of`, a backend SHOULD return stable results. Backends that are inherently nondeterministic declare it via `stats().extra["nondeterministic"] = True`; the harness then averages over N runs and reports variance.
5. **Honest accounting.** `ingest`/`query`/`consolidate` return real `Usage`; the harness independently wall-clocks `latency_ms`. Backends must not under-report tokens/cost.
6. **No gold access.** A backend never receives gold labels, the category tag, or the expected answer — only `prompt`, `k`, `as_of`.

## 5. Baseline backends (shipped in-box)

Every score is meaningless in isolation; these references make it relative. All three implement §4 with no special-casing.

| Backend | Behavior | Why it exists |
|---|---|---|
| **NoMemory** | `ingest` no-op; `query` returns empty (and a "don't know" answer). | The floor. Any real memory must beat it; if it doesn't, the harness or backend is broken. |
| **LongContext** | Stores raw sessions; `query` concatenates all sessions with `timestamp <= as_of` (most-recent-first) up to a token budget `B`, then lets an LLM answer. No retrieval/ranking. | The "just stuff the window" strawman. Exposes cost/latency blow-up and storage growth as sessions accumulate, and hits a ceiling once history exceeds `B`. |
| **NaiveRAG** | Chunk sessions → embed → vector index; `query` embeds the prompt and returns top-k by cosine. No contradiction handling, no curation. | The standard reference. **Must** show `staleness > 0` on the contradiction suite — it has no mechanism to prefer fresh facts. |

Two further adapters ship as **documented stubs** (interface-complete, raise `NotImplementedError` until wired):

- **Letta** — third-party system; the canonical "external backend" integration path.
- **CuratedBrain** — the system-under-test; the backend the scoreboard ultimately judges.

## 6. Test categories (v1)

Each category defines: what it probes, the scenario shape, how gold labels are constructed, and its primary metrics.

1. **Longitudinal recall.** A fact is planted in an early session and queried many sessions later with no reinforcement. *Gold:* the originating fact's item ID. *Probes:* does it persist and surface? *Primary:* recall@k.
2. **Contradiction handling.** A fact is planted, then updated in a later session (creating a superseded version). A query after the update must return the **current** value and not the stale one. *Gold:* current fact ID + the set of superseded IDs that must NOT win. *Primary:* contradiction-resolution accuracy, staleness.
3. **Multi-hop retrieval.** The answer requires combining ≥2 facts planted in *different* sessions (e.g., A in s3, B in s12). *Gold:* the full supporting set {A, B}. *Primary:* recall of the complete support set, answer accuracy.
4. **Recency-vs-relevance tradeoff.** A highly relevant-but-old item competes with a recent-but-marginal one. Correct behavior depends on explicit **query intent** ("what's my current X?" ⇒ recency; "what did I say about X back then?" ⇒ relevance). *Gold:* the intent-appropriate item. *Primary:* precision@k, ranking quality.
5. **Needle-across-sessions.** One distinctive fact is buried among many sessions of distractor content — the cross-session analog of needle-in-haystack. *Gold:* the needle item ID. *Primary:* recall@k at depth (as #sessions grows).

## 7. Metrics — the scorecard (7)

All seven are emitted for every `(backend × suite)` run, broken out per category and overall. Let `R_k` = retrieved top-k item set, `G` = gold supporting-item set for a query.

1. **recall@k** = `|R_k ∩ G| / |G|`. Did the right items surface?
2. **precision@k** = `|R_k ∩ G| / |R_k|`. How much of what surfaced was relevant?
3. **contradiction-resolution accuracy** = fraction of contradiction queries that *pass*, where pass ≙ the current fact ∈ `R_k` **and** no superseded version of it appears in `R_k`. (When the backend returns an `answer`, an LLM judge additionally checks the answer reflects the current value, not the stale one.)
4. **staleness** = `(# returned items that are superseded as_of) / (# returned items)`, averaged over staleness-probe queries. Lower is better; this is the metric naive-RAG should fail.
5. **query latency** = wall-clock ms per `query`, reported as p50 / p95 / p99.
6. **cost per query** = USD and tokens per query, with amortized ingest cost (`ingest_usd / num_queries`) reported separately from query-time cost.
7. **storage growth over time** = `stats().bytes` and `item_count` sampled as a function of #sessions ingested; reported as footprint at N sessions and growth slope.

**Pass/fail vs. reported.** Metrics 1–4 are *quality* metrics and participate in determinism/regression checks. Metrics 5–7 are *operational* and environment-dependent: they are always reported and compared, but excluded from the determinism hash (§10.3) so identical-quality runs aren't flagged by latency jitter.

## 8. Synthetic data generation + ground truth

The defining property: **ground truth is constructed, not inferred.** Gold labels come from a structured layer the generator controls — never from an LLM judging text — so labels are exact.

- **Structured layer.** The generator plants `Fact(entity, attribute, value, valid_from)` objects on a timeline, schedules updates (creating superseded versions), wires multi-hop chains, and sprinkles distractor facts. Each fact gets a stable `item_id`.
- **Surface layer (optional).** An LLM may render natural-language turns *around* the structured facts to make sessions realistic. Renders are cached keyed by `(seed, fact_id)` so they're reproducible and don't re-incur cost. **Labels never depend on the rendered text.**
- **Query derivation.** For each category, queries are generated from the structured layer with: gold answer, gold supporting-item set, `as_of`, intent tag (for category 4), and category tag.
- **Generator parameters:** `num_sessions`, `turns_per_session`, `fact_density`, `distractor_ratio`, `update_rate`, `multi_hop_chains`, `needle_depth`, `seed`.
- **Determinism:** fixed `seed` ⇒ identical scenario + labels; the harness asserts this via a content hash.
- **Versioning:** each suite has a `dataset_version`; changing generation logic bumps it and is recorded in the scorecard.

## 9. Real conversation logs — the path

v1 defines the path and ships the entry point; full real-log suites are roadmap.

- **Importer.** Maps external transcripts (e.g., exported chat logs) into the `Session` schema with timestamps. Shipped and tested in v1.
- **Labeling.** Two tiers: (a) a **manual annotation format** (YAML sidecar: query → gold answer + supporting item IDs + as_of + category) for high-confidence gold; (b) an optional **LLM-assisted label proposal** that pre-fills the sidecar but is explicitly flagged low-confidence for human review.
- **v1 deliverable:** the importer + schema + one small hand-annotated real-log sample that runs end-to-end through the same metrics. Building large labeled real-log suites is explicitly roadmap.

## 10. Runner, CLI, reproducibility & scorecard format

### 10.1 CLI

```
mem-eval run --backend naive_rag --suite v1 --seed 42 --k 10 --out results/
mem-eval compare results/*.json          # side-by-side scoreboard across backends
mem-eval gen   --suite v1 --seed 42 --out datasets/   # materialize a dataset
```

### 10.2 Scorecard JSON (shape)

```json
{
  "run_id": "…",
  "backend":  {"name": "naive_rag", "version": "0.3.1"},
  "dataset":  {"suite": "v1", "version": "2026.6", "seed": 42},
  "config":   {"k": 10, "consolidate_cadence": "per_session"},
  "metrics":  {
    "overall": {"recall_at_k": 0.0, "precision_at_k": 0.0,
                "contradiction_resolution_accuracy": 0.0, "staleness": 0.0,
                "latency_ms": {"p50": 0, "p95": 0, "p99": 0},
                "cost_per_query_usd": 0.0,
                "storage": {"bytes_at_n": {}, "growth_slope": 0.0}},
    "by_category": { "longitudinal_recall": {…}, "contradiction": {…}, … }
  },
  "env": {"git_sha": "…", "model": "…", "embedding_model": "…", "timestamp": "…"},
  "determinism_hash": "…"
}
```

### 10.3 Reproducibility

- A run records `backend.version`, `dataset.version`, `seed`, `config`, and model/embedding versions.
- The **determinism hash** is computed over quality metrics (1–4) + dataset identity. Two runs with the same seed/version/config MUST produce the same hash. Operational metrics (5–7) are reported but excluded from the hash.
- Nondeterministic backends (declared per §4.3.4) run N times; the harness reports mean and variance, and the determinism check is relaxed to "within tolerance."

## 11. Acceptance criteria (v1 definition of done) — falsifiable

These are the criteria the `pytest` gate encodes. Each is machine-checkable.

- **A1 — Contract + baselines run.** The adapter interface is defined and frozen; `NoMemory`, `LongContext`, and `NaiveRAG` all implement it and run end-to-end on the v1 synthetic suite with no error.
- **A2 — Coverage with exact labels.** Each of the 5 categories has ≥1 generator producing scenarios + exact gold labels, deterministic under a fixed seed.
- **A3 — All metrics emitted.** All 7 metrics appear in the scorecard JSON for every `(backend × suite)` run, per-category and overall.
- **A4 — Discrimination (the crux).** On the v1 suite: `recall@k(NoMemory) < recall@k(NaiveRAG)`; `staleness(NaiveRAG, contradiction suite) > 0`; and `contradiction_resolution_accuracy(NoMemory) ≈ 0 < NaiveRAG`. If the harness cannot separate these known-different baselines, it fails — full stop.
- **A5 — Determinism.** Two runs with identical seed/version/config produce an identical determinism hash.
- **A6 — Comparison.** `mem-eval compare` emits a single scoreboard across ≥2 backends from their scorecard JSONs.
- **A7 — Gate.** `pytest -q` encodes A1–A6 and exits 0, with metrics and generators under test.

## 12. Proposed repo structure

```
src/mem_eval/
  adapters/        base.py        # §4 — the frozen contract
                   no_memory.py
                   long_context.py
                   naive_rag.py
                   letta.py        # stub
                   curated_brain.py# stub (system under test)
  data/            schema.py      # Session/Turn/Fact/Scenario/Query
                   generators/    # one per category + shared planting engine
                   render.py      # optional LLM surface layer (cached)
                   importers/     # real-log → Session
  metrics/         recall.py precision.py contradiction.py staleness.py
                   latency.py cost.py storage.py
  runner/          cli.py orchestrate.py
  report/          scorecard.py   # JSON emit + markdown compare
tests/             # acceptance suite encoding A1–A7
datasets/          # materialized, versioned fixtures
results/           # scorecard outputs
PRD.md
CLAUDE.md
PROGRESS.md        # written only if the gate can't pass (see CLAUDE.md)
```

## 13. Build stages / roadmap

Mirrors the Workflow in `CLAUDE.md`; each stage ends with the gate green + an Opus 4.8 reviewer pass + one commit.

1. **Contract + scaffold + acceptance suite.** Write `adapters/base.py`, the data schema, and the `pytest` acceptance tests from §11 (the gate itself). Tests fail until later stages fill them in.
2. **Baselines.** Implement `NoMemory`, `LongContext`, `NaiveRAG` against the contract; leakage/isolation probes pass.
3. **Generators.** All 5 category generators + structured gold + determinism (A2).
4. **Metrics + scorecard.** All 7 metrics + JSON/markdown emit (A3).
5. **Runner end-to-end.** `run` + `compare` wired; discrimination (A4) and determinism (A5/A6) pass on the v1 suite.
6. **Real-logs path + stubs.** Importer + annotation format + one annotated sample; Letta/CuratedBrain stubs interface-complete; docs.

**Post-v1 roadmap:** large labeled real-log suites; hosted leaderboard; more backends; adversarial/curation-stress scenarios; cost/latency budget regression alerts.

## 14. Risks & open questions

- **Surface-layer nondeterminism/cost.** LLM-rendered turns can drift and cost money → mitigated by caching renders by `(seed, fact_id)` and keeping gold in the structured layer.
- **Model/embedding version drift** changes scores → pin and record versions in every scorecard; comparisons must hold env constant.
- **"Relevance" is subjective**, especially for recency-vs-relevance → resolved by explicit query-intent tags rather than implicit judgment.
- **Operational metrics are environment-dependent** → reported and compared but excluded from pass/fail determinism; fair comparison requires the same machine/config.
- **Real-log labeling is expensive** → v1 ships the path + a small sample, not a large suite.
- **Open:** default `k`, token budget `B` for LongContext, and `consolidate` cadence — set sensible defaults in Stage 2 and expose via config.

---

*End of PRD. The single command that proves this build is healthy is `pytest -q` against the §11 acceptance suite.*

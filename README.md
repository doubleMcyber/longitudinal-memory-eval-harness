# Longitudinal Memory Eval Harness (LMEH)

A reproducible benchmark + runner that measures how well an agent's **memory
system retains and uses information across sessions** — persistence,
contradiction handling, and curation quality — the failure modes that
single-context "needle in a haystack" tests cannot see.

Plug a memory backend into one frozen adapter interface and get a deterministic
**scorecard**: recall@k, precision@k, contradiction-resolution accuracy,
staleness, query latency, cost per query, and storage growth — per category and
overall, with bootstrap confidence intervals.

> Status: the `pytest -q` acceptance gate (PRD §11, A1–A7) is green. See
> [`PRD.md`](PRD.md) for the spec and [`PRODUCTION_ROADMAP.md`](PRODUCTION_ROADMAP.md)
> for the path to a labs-grade benchmark.

## Why it discriminates

Every score is relative to in-box reference points, and the scoreboard is a
**meaningful spectrum** — it rewards good behavior, not just punishes nothing:

| backend | recall@k | contradiction_acc | staleness | what it is |
|---|---|---|---|---|
| `no_memory` | 0.00 | 0.00 | 0.00 | the floor (stores nothing) |
| `naive_rag` | ~0.91 | ~0.2 (varies) | high | standard cosine RAG, no curation |
| `temporal_rag` | 1.00 | ~0.6–0.8 (varies, **< 1.0**) | low | contradiction-aware reference |

The separation is **emergent and not tuned to the reference**. The contradiction
suite is difficulty-stratified and RNG-driven across two axes — crowding/chain
length and *phrasing* (possessive vs coreference). `naive_rag`'s accuracy varies
by seed (no supersession mechanism); `temporal_rag` resolves supersession via
honest, text-only consolidation (no gold access) but **cannot** resolve
adversarial coreference phrasing, so its score is strictly below 1.0 with visible
**headroom** for a real system to claim. Anchored scenarios guarantee
`no_memory < naive_rag < temporal_rag < 1.0` for every seed.

## Install & run

Pure standard library + `pytest`. No network, no model downloads required.

```bash
# run the acceptance gate
pip install pytest && pytest -q          # reads pythonpath=src from pyproject.toml

# CLI
PYTHONPATH=src python -m mem_eval.runner.cli run --backend temporal_rag --suite v1 --seed 42 --k 10 --out results/
PYTHONPATH=src python -m mem_eval.runner.cli compare results/*.json          # side-by-side scoreboard (with 95% CIs)
PYTHONPATH=src python -m mem_eval.runner.cli gen   --suite v1 --seed 42 --scale large --out datasets/
PYTHONPATH=src python -m mem_eval.runner.cli depth --backend naive_rag --depths 8,16,32,64
PYTHONPATH=src python -m mem_eval.runner.cli iaa   --transcript T.json --annotations multi.json
```

## Test categories (5) and metrics (7)

Categories (PRD §6): longitudinal recall, contradiction handling, multi-hop
retrieval, recency-vs-relevance, needle-across-sessions.

Metrics (PRD §7): recall@k, precision@k, contradiction-resolution accuracy,
staleness, query latency (p50/p95/p99), cost per query (+ amortized ingest),
storage growth. Quality metrics (1–4) carry bootstrap 95% CIs and feed the
determinism hash; operational metrics (5–7) are reported but excluded from it.

## Adding a backend (the frozen contract)

Implement `mem_eval.adapters.base.MemoryBackend` — and nothing else:

```python
def reset(self) -> None: ...                       # wipe state (isolation)
def ingest(self, session: Session) -> Usage: ...   # persist a session (timestamp order)
def query(self, prompt, k, as_of) -> QueryResult:  # retrieve up to k items usable to answer
def consolidate(self) -> Usage: ...                # optional curation hook (default no-op)
def stats(self) -> BackendStats: ...               # footprint + item count
```

Contract rules are enforced by tests (PRD §4.3): time discipline (`as_of`),
provenance (`source_session`/`timestamp`), isolation, determinism, honest usage,
and **no gold access** — a backend sees only `prompt`, `k`, `as_of`. Register it
in `mem_eval.adapters` and you get a graded scorecard.

### Plugging in real models

- **Embeddings**: `ConfigurableRAG(embedding=...)` takes any `EmbeddingFn`
  (`embed`/`similarity`). Ships `TokenCosineEmbedding` (offline default) and
  `HashEmbedding`; `SentenceTransformerEmbedding`/`OpenAIEmbedding` are documented
  integration stubs.
- **Answer judging**: `LLMJudge(complete_fn=..., model=...)` adjudicates semantic
  equivalence with your model; the offline default `NormalizingJudge` is
  deterministic (whole-token containment + numeric/date/alias tolerance).

## Reproducibility

Same seed + scale + backend version + config ⇒ identical `determinism_hash`
(computed over the four quality metrics + dataset identity; operational metrics
excluded). Every scorecard records a manifest: backend/version, dataset
(suite/version/seed/scale), config, env (harness/python/platform/embedding/judge),
and the determinism hash.

## Real conversation logs

Import external transcripts into the `Session` schema, attach a manual annotation
sidecar (JSON; YAML supported when PyYAML is present), and run them through the
*same* metrics. Multi-annotator gold is vetted with **inter-annotator agreement**
(Cohen's κ on support membership + answer agreement) before admission.

## Repo layout

```
src/mem_eval/
  adapters/   base.py (frozen contract) + no_memory/long_context/naive_rag
              temporal_rag (reference) + configurable_rag/embeddings (real path)
              letta/curated_brain (stubs)
  data/       schema.py, suite.py (scale presets), generators/, importers/
  grading/    judge.py (answer grading), agreement.py (IAA)
  metrics/    recall/precision/contradiction/staleness/latency/cost/storage + significance
  runner/     orchestrate.py, cli.py, analyze.py (depth)
  report/     scorecard.py (JSON + markdown compare + determinism hash)
tests/        acceptance suite (A1–A7) + per-feature tests
```

## License & status

Research benchmark, v0.2. The single command that proves the build is healthy is
`pytest -q`.

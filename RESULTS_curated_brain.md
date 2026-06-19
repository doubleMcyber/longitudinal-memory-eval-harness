# Preliminary result — Curated Brain vs the shipped references

**Suite:** `v1`, scale `standard`, seed `42`, `k=10` · **fully offline & deterministic**
(determinism hash `bacda629…`, identical across runs) · harness `compare` output below.
Reproduce: `for b in curated_brain temporal_rag naive_rag semantic_rag long_context no_memory; do
PYTHONPATH=src python -m mem_eval.runner.cli run --backend $b --suite v1 --seed 42 --k 10 --scale standard --out results/; done && PYTHONPATH=src python -m mem_eval.runner.cli compare results/*.json`

| metric | curated_brain | temporal_rag | naive_rag | semantic_rag | long_context | no_memory |
| --- | --- | --- | --- | --- | --- | --- |
| recall@k | 0.760 | **0.920** | 0.800 | 0.920 | 0.600 | 0.000 |
| precision@k | **0.630** | 0.536 | 0.391 | 0.403 | 0.188 | 0.000 |
| contradiction_acc | **0.800** | **0.800** | 0.100 | 0.100 | 0.100 | 0.000 |
| staleness (lower=better) | 0.200 | **0.060** | 0.280 | 0.280 | 0.280 | 0.000 |
| answer_acc | 0.640 | **0.760** | 0.680 | 0.760 | 0.680 | 0.000 |
| cost_per_query_usd | **~0.0000** | 0.0001 | 0.0001 | 0.0001 | 0.0004 | 0.000 |
| storage_growth_slope | 516.96 | 45.72 | 45.72 | 45.72 | 45.72 | 0.00 |

## Honest verdict: **not a clean win** (yet)

The headline bar — *Curated Brain ≥ temporal_rag on recall AND contradiction at ≤ cost* — is
**not met**: CB **loses recall (0.76 vs 0.92)**. What CB *does* show, on a neutral third-party
harness it never saw:

- **Best precision of any backend (0.63)** and **lowest cost** (no LLM; structured store).
- **Ties the contradiction-aware reference on contradiction-resolution (0.80)**, far above the
  RAG backends (0.10) — its bi-temporal supersede genuinely works.
- **Wins long-range recall by category (0.83 vs 0.67)** — the curation thesis on its home turf.

## Where it loses, and why (per-category, all GENERAL gaps — not adapter bugs)

Verified by an independent adversarial review: the adapter is contract-clean (17/17 contract
tests), faithful (72 citations, **0 unmapped, 0 gold turns wrongly excluded**), and does not
peek at gold. So the losses are real capability gaps:

| category | CB recall | TR recall | cause |
| --- | --- | --- | --- |
| longitudinal_recall | **0.83** | 0.67 | CB wins |
| needle | 1.00 | 1.00 | tie |
| multi_hop | 0.67 | 1.00 | CB cites only the **final hop's turn**, missing chain turns — though CB's multi_hop *answer* (0.33) beats TR's (0.00) |
| recency_relevance | 0.50 | 1.00 | CB has no **"originally/first" history-intent** path; returns the old value for both intents |
| contradiction | 0.80 | 1.00 | heuristic extractor doesn't always canonicalize old/new phrasings to the same `(subject, predicate)`, so supersede misfires → recall + staleness hit |

## Fair-comparison disclosures

- **Answer convention:** CB returns a **top-1 answer** (its resolved fact / best line), matching
  the baselines' single `items[0].content`. (An earlier multi-candidate answer was worth +0.08
  answer_acc; removed for apples-to-apples — CB still loses answer_acc either way.)
- **Storage slope (517 vs 46) is a serialization artifact**, not real bloat: `stats().bytes` =
  `len(cb.snapshot())`, a verbose JSON dump of facts + provenance + embedding config (~790 B/item)
  vs the references' ~46 B/item raw chunks. Not claimed as a CB win or a real loss.

## The general fixes that would close the gap (next, NOT benchmark-tuning)

1. **Multi-hop provenance:** cite every fact in `resolve_path`'s chain, not just the final one
   (CB already traverses them) → multi_hop recall → ~1.0.
2. **History/recency intent:** a planner path for "originally/first/initially" that resolves the
   earliest fact, and recency disambiguation for "current/now" → recency_relevance up.
3. **Extractor canonicalization robustness** on contradiction phrasings → contradiction recall
   + staleness.
4. **Storage accounting:** measure the live store, not the full snapshot blob.

Per the plan's stop rule, we report this honestly rather than tune to the benchmark. CB is
**competitive, far cheaper, more precise, and contradiction-strong**, with a clear, general
roadmap to overtake the reference on recall.

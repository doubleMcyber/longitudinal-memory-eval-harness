# Production Roadmap — from passing gate to a benchmark labs run

The v1 harness proves the *mechanism* works (deterministic, reproducible, the gate
discriminates known-different baselines). To be a benchmark top AI labs bench on,
it must also be *meaningful, hard, emergent, and operationally rigorous*. This
roadmap tracks that elevation. Each wave keeps `pytest -q` green and commits once.

## Rubric (definition of production-ready)
1. **Meaningful spectrum** — separates floor → naive → a *good* reference, not just bad-from-nothing.
2. **Emergent discrimination** — arises from backend mechanism on hard, varied data; not hardcoded constants.
3. **Non-trivial retrieval** — paraphrase + heavy distractors; naive lexical matching cannot ace it.
4. **Pluggable real backends** — injectable embedding fn + LLM answer/judge; tested wiring.
5. **Contradiction-aware reference** — a backend that genuinely resolves supersession (no gold access).
6. **Scale & depth** — configurable suite size; depth-scaling curves (recall vs #sessions).
7. **Robust answer grading** — normalization, numeric/alias/date tolerance, pluggable judge.
8. **Statistical rigor** — bootstrap confidence intervals; score deltas are trustworthy.
9. **Real-log IAA** — multi-annotator format + inter-annotator agreement (Cohen's κ).
10. **Operational polish** — README, CI, packaging, reproducibility manifest; determinism preserved.

## Waves
- **Wave 1 — Meaningful + emergent (the headline).** `TemporalRAG` contradiction-aware reference;
  RNG-driven, difficulty-stratified, multi-update contradiction data; harder distractors.
  Tests assert the *ordering* NoMemory < NaiveRAG < TemporalRAG (inequalities, multi-seed), not constants.
- **Wave 2 — Real backends + robust grading.** Pluggable `EmbeddingFn` (offline default; sentence-transformers/
  OpenAI hooks) via `ConfigurableRAG`; `AnswerJudge` protocol with normalizing default + LLM-judge adapter.
- **Wave 3 — Scale/depth + statistics.** Configurable generator scale via CLI/config; depth-scaling report;
  bootstrap confidence intervals surfaced in scorecard + compare.
- **Wave 4 — Real-log IAA + ops.** Multi-annotator sidecars + Cohen's κ; README, CI workflow, packaging,
  reproducibility manifest, dataset versioning.

Each wave: implement → independent verification subagent (up to spec?) → commit.
After all waves: planning subagent (what next) → main-agent self-check against this rubric.

## Cycle 1 self-check (after waves 1–4)
Met: meaningful spectrum, emergent contradiction, pluggable backends, contradiction-aware
reference, scale/depth, robust grading, bootstrap CIs, real-log IAA, ops polish. **Not yet**
production-ready — a planning pass surfaced credibility-gating gaps:
- **D1** retrieval is lexical token-cosine on near-verbatim data (grep wins; semantic seam unproven).
- **D2** TemporalRAG scores a flat 1.0 because contradiction data is co-designed with its `topic_key`
  heuristic — looks tuned to make its own reference win.
- **D3** real-log suite is one tiny sample; no external-validity study.
- **D4** no curation-stress / poisoning scenarios; `consolidate()` never bites.
- **D5** modeled cost/latency rendered beside quality without a "modeled" caveat.

## Cycle 2 (DONE — closed the cardinal credibility issues + curation pillar)
- **Wave 5 (D2 + D5) ✅**: adversarial coreference phrasing defeats `topic_key`; TemporalRAG contradiction
  is emergent and < 1.0 with headroom; NaiveRAG < TemporalRAG guaranteed. Modeled ops labeled.
- **Wave 6 (D1) ✅**: structural lexical gap (token-cosine recall = 0 across 120 seeds) + offline semantic
  embedding (SynonymHashEmbedding) + SemanticRAG; semantic provably beats lexical — the seam is meaningful.
- **Wave 7 (D4) ✅**: real `TemporalRAG.consolidate()` (exact-duplicate compaction, time-discipline safe) +
  curation-stress test; curated storage stays flat (slope ~1.4) vs naive (slope 55). The curation pillar is live.

The scoreboard is now genuinely multi-axis: **semantic_rag** wins retrieval, **temporal_rag** wins
curation+contradiction (with headroom < 1.0), neither dominates — a real system must do all three.

## Cycle 3 (DONE — external validity)
- **D3 ✅**: a meaningfully-sized real-log corpus (`corpora/real_logs_v1/`: 8 hand-authored scenarios,
  27 sessions, 9 queries, 3 annotators, κ ≈ 0.88) + an **external-validity study** (`mem_eval/validity/`,
  `mem-eval validity`). A backend panel is ranked on the synthetic suite and on the real corpus by a
  transparent quality composite; the harness reports **Spearman ρ** (headline), Kendall τ, and an exact
  permutation p, with a per-axis decomposition. On the shipped data the rankings are **identical**
  (temporal_rag wins both, no_memory floors both): **ρ = 1.0, τ = 1.0, p ≈ 0.017**, with positive transfer
  on every axis (recall .92, precision .97, contradiction .79, answer 1.0). Rank-correlation primitives
  (`mem_eval/metrics/correlation.py`: tie-averaged Spearman, Kendall τ-b, exact permutation p) are pure
  stdlib + deterministic.

  **Scope (honest):** the corpus is a *curated, hand-authored stand-in* for human-collected production
  logs; the study is the reusable instrument and a genuine labeled corpus of the same format is a drop-in
  replacement. The validity claim is bounded: to the extent real logs exercise these capabilities, the
  synthetic ranking predicts the real one.

## Remaining (next cycle — adoption-scale, not credibility-gating)
- **D3-scale**: replace the curated stand-in corpus with a *genuinely human-collected*, large, IAA-vetted
  real-log suite (the format/importer/study already ship; this is a data-collection + human-labeling effort).
- **Planner item 7**: give multi_hop / recency_relevance the same RNG-stratified emergent treatment.
- Hosted leaderboard + tamper-checked submission format (held-out seeds).

## Carry-forward notes (from verification subagents)
- **W1**: TemporalRAG scores a flat 1.0/0.0 on contradiction — the data never stresses its
  consolidation heuristic. A later "hard data" wave should add chains that challenge `topic_key`
  (coreference/pronouns, non-copula phrasing) so the *reference* score becomes non-constant too,
  mirroring what emergence did for the baseline.

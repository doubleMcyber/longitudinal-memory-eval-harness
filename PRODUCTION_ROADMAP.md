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

## Carry-forward notes (from verification subagents)
- **W1**: TemporalRAG scores a flat 1.0/0.0 on contradiction — the data never stresses its
  consolidation heuristic. A later "hard data" wave should add chains that challenge `topic_key`
  (coreference/pronouns, non-copula phrasing) so the *reference* score becomes non-constant too,
  mirroring what emergence did for the baseline.

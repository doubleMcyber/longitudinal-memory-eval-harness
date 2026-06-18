# real_logs_v1 — real-style conversation corpus (D3)

A meaningfully-sized, hand-authored conversation corpus used by the harness's
**external-validity study** (`mem-eval validity`). It is the real-log counterpart
to the synthetic suite: independently written natural-language logs with
hand-labeled gold, run through the *same* runner and metrics.

## What this is (and is not)

- **Is**: a curated, hand-authored *stand-in* for human-collected production logs.
  8 scenarios, 27 sessions, 9 queries, spanning the five PRD categories
  (contradiction, longitudinal recall, multi-hop, needle-in-haystack,
  recency-vs-relevance). Two of the longitudinal-recall scenarios (`rl-reside`,
  `rl-employer`) additionally embed a **lexical-gap / paraphrase** challenge — the
  query and the supporting turn share meaning but no content tokens — which is the
  seam that separates a semantic retriever from a lexical one. The surface text —
  entities, phrasings, distractors — is independent of the synthetic generators.
- **Is not**: genuinely human-*collected* logs. The honest claim the study makes
  is bounded accordingly (see below). A real human-collected, IAA-vetted corpus of
  this same format is a **drop-in replacement** — every line of the importer and
  study module is reused unchanged.

## The defining rule (same as the synthetic suite)

**Ground truth is hand-labeled in the structured layer, never inferred from text.**
Gold lives only in `support` / `superseded` turn references + the gold answer
string. No backend ever sees this directory; the transcript text it ingests
carries no labels.

## Files

| file | what it holds |
|---|---|
| `corpus.json` | scenarios: `sessions[]` (transcript) + `queries[]` (prompt, `as_of`, gold answer, `support`/`superseded` refs, optional `intent`/`answer_aliases`) |
| `annotators.json` | 3 independent annotators' `labels[]` (per-query `gold_answer` + `support` refs) for inter-annotator agreement |

## Labeling protocol

1. **High-confidence gold (tier a).** A turn is a *support* ref iff it is the turn
   a human would point to as evidence for the answer; a *superseded* ref is an
   earlier, now-stale assertion of the same fact that a correct system must NOT
   surface as current.
2. **Answers** are written as a token-subsequence of their support turn, so
   provenance-grading and answer-grading agree.
3. **Adjudication.** `corpus.json` holds the adjudicated gold; `annotators.json`
   holds the raw independent labels used to measure agreement.

## Inter-annotator agreement

```bash
python -c "from mem_eval.data.importers.transcript import corpus_iaa; print(corpus_iaa())"
# support_kappa ~0.88 (high, <1.0 — two defensible disagreements), answer_agreement 1.0
```

Cohen's κ is computed on the binary "is this turn supporting?" decision over each
scenario's candidate turns, pooled across scenarios; answer agreement is exact
(normalized) match. It covers the 6 single-support queries the three annotators
labeled in common (the remaining multi-hop and recency-vs-relevance queries are
not in the IAA subset).
Low agreement would flag an ambiguous query or guideline before the gold is
trusted.

## How it is used (external validity)

`mem-eval validity` ranks a backend panel on the synthetic suite and on this
corpus by a quality composite, then reports Spearman ρ / Kendall τ / permutation
p between the two rankings (+ per-axis decomposition). On the shipped data the
orderings are identical (ρ = 1.0): evidence that the synthetic ranking predicts
the real one **to the extent real logs exercise these capabilities** — a bounded,
conditional claim, not a guarantee about arbitrary production traffic.

## Swapping in a genuine corpus

Replace `corpus.json` (same schema) and `annotators.json` with human-collected,
human-labeled data, then re-run `mem-eval validity --corpus path/to/corpus.json`.
Nothing else changes. Growing this into a large, genuinely-collected, IAA-vetted
suite is the documented next step (PRODUCTION_ROADMAP.md).

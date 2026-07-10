# Pre-registration — preference-path frozen re-run (written before the run)

Date frozen: 2026-07-10, before any benchmark execution or result look.

## Hypothesis

CB's preference fact family (library commit `7a3a777`, opt-in) plus the `CB_PREF=1` adapter
mapping (this repo, this commit) improves answer accuracy on the LongMemEval
`single-session-preference` category, where CB measured 0.087 (stratified n=23, 2026-07-02)
vs Letta 0.217.

## Frozen config

- Dataset: `data/longmemeval_oracle`, ALL 30 `single-session-preference` questions
  (`--type single-session-preference`), seed 42.
- Model: local Ollama `qwen2.5:7b` for extraction/answering/judging; `nomic-embed-text`
  embedder — identical in both arms; same machine, serialized runs.
- Arm A (baseline): CB backend at library HEAD `a967291`+WS9 docs, `CB_PREF` unset.
- Arm B (lever): identical except `CB_PREF=1` (adapter maps LLM-extracted favorite/likes
  attribute forms into `preference:<topic>` facts; the library's schema-driven preference
  aggregation then fires on preference questions).
- Runner: `bench_longmemeval.py`, per-question fresh backend, crash-safe checkpoints.

## Pre-registered acceptance rule (the dates.py / temporal-lever precedent)

Keep the adapter lever only on a REAL lift in preference answer accuracy (more questions
gained than lost, and a delta larger than one question, i.e. >= +2/30). On a null or negative
result: revert the adapter lever, keep the library capability (it is opt-in and shipped on its
own merits), and record the honest negative in PROGRESS.md and RESULTS_longmemeval.md.
No second attempt tuned against these numbers in this session.

## Blindness statement

Gate A (diagnostic suite) was re-verified byte-identical (hash 673a25c7) at library commit
`7a3a777` before this file was written. No LongMemEval preference numbers have been looked at
since the 2026-07-02/03 runs recorded in RESULTS_longmemeval.md.

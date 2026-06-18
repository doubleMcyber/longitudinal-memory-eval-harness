"""End-to-end run orchestration (PRD §10, A1/A3/A5).

Runs one backend over a suite with full contract isolation (reset before each
scenario, ingest in timestamp order), produces per-query EvalRecords by resolving
returned items to gold via provenance, computes all 7 metrics, and assembles the
scorecard. The harness independently wall-clocks query latency (PRD §4.3.5).
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from time import perf_counter

from mem_eval import __version__ as HARNESS_VERSION
from mem_eval.adapters.base import MemoryBackend, Usage
from mem_eval.data.schema import Suite, superseded_as_of
from mem_eval.grading.judge import DEFAULT_JUDGE, AnswerJudge
from mem_eval.metrics import compute_metrics_block, storage_growth
from mem_eval.metrics.records import EvalRecord
from mem_eval.report.scorecard import build_scorecard

DEFAULT_K = 10
DEFAULT_CADENCE = "per_session"
MODEL = "offline-deterministic-extractor"
EMBEDDING_MODEL = "offline-token-cosine"


def _consolidate_if(backend: MemoryBackend, when: str, cadence: str) -> Usage:
    if cadence == when:
        return backend.consolidate()
    return Usage()


def _evaluate_scenario(backend: MemoryBackend, scenario, k: int, cadence: str, judge: AnswerJudge):
    backend.reset()
    ingest = Usage()
    for sess in sorted(scenario.sessions, key=lambda s: s.timestamp):
        ingest = ingest + backend.ingest(sess)
        ingest = ingest + _consolidate_if(backend, "per_session", cadence)
    ingest = ingest + _consolidate_if(backend, "end", cadence)

    sup = {}  # as_of -> superseded set (cached per distinct as_of)
    records: list[EvalRecord] = []
    for q in scenario.queries:
        kk = q.k or k
        t0 = perf_counter()
        res = backend.query(q.prompt, kk, q.as_of)
        latency_ms = (perf_counter() - t0) * 1000.0

        resolved = tuple(scenario.resolve_item(it) for it in res.items)
        if q.as_of not in sup:
            sup[q.as_of] = superseded_as_of(scenario, q.as_of)
        superseded_returned = sum(1 for fid in resolved if fid in sup[q.as_of])
        answer_correct = judge.judge(res.answer, q.gold_answer, q.answer_aliases)

        records.append(
            EvalRecord(
                query_id=q.query_id,
                category=q.category,
                intent=q.intent,
                gold_support=frozenset(q.gold_support),
                gold_superseded=frozenset(q.gold_superseded),
                retrieved=resolved,
                retrieved_count=len(res.items),
                superseded_returned=superseded_returned,
                answer=res.answer,
                gold_answer=q.gold_answer,
                answer_correct=answer_correct,
                latency_ms=latency_ms,
                usage=res.usage,
            )
        )
    return records, ingest


def _storage_samples(backend: MemoryBackend, scenario, cadence: str):
    backend.reset()
    sessions = sorted(scenario.sessions, key=lambda s: s.timestamp)
    n = len(sessions)
    checkpoints = sorted({1, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), n})
    samples = []
    for i, sess in enumerate(sessions, start=1):
        backend.ingest(sess)
        _consolidate_if(backend, "per_session", cadence)
        if i in checkpoints:
            st = backend.stats()
            samples.append((i, st.bytes, st.item_count))
    return samples


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_eval(
    backend: MemoryBackend,
    suite: Suite,
    *,
    k: int = DEFAULT_K,
    consolidate_cadence: str = DEFAULT_CADENCE,
    judge: AnswerJudge | None = None,
    timestamp: str | None = None,
) -> dict:
    judge = judge or DEFAULT_JUDGE
    all_records: list[EvalRecord] = []
    ingest_total = Usage()
    for scenario in suite.scenarios:
        recs, ingest = _evaluate_scenario(backend, scenario, k, consolidate_cadence, judge)
        all_records.extend(recs)
        ingest_total = ingest_total + ingest

    # storage growth: sample on the deepest scenario (most sessions)
    deepest = max(suite.scenarios, key=lambda s: len(s.sessions))
    storage = storage_growth(_storage_samples(backend, deepest, consolidate_cadence))

    metrics_block = compute_metrics_block(all_records, ingest_total.usd, storage)

    backend_meta = {"name": backend.name, "version": backend.version}
    nd = bool(backend.stats().extra.get("nondeterministic", False))
    backend_meta["nondeterministic"] = nd
    dataset_meta = {
        "suite": suite.suite,
        "version": suite.dataset_version,
        "seed": suite.seed,
        "scale": suite.config_name,
    }
    config = {"k": k, "consolidate_cadence": consolidate_cadence}
    env = {
        "git_sha": _git_sha(),
        "harness_version": HARNESS_VERSION,
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "answer_judge": judge.name,
        "judge_model": getattr(judge, "model", None),
        "pricing_model": "token-rate-v1 (modeled, not measured)",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
    }
    return build_scorecard(
        backend=backend_meta,
        dataset=dataset_meta,
        config=config,
        metrics_block=metrics_block,
        env=env,
    )


__all__ = ["run_eval", "DEFAULT_K", "DEFAULT_CADENCE"]

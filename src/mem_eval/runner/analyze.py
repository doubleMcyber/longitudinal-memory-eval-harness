"""Depth-scaling analysis (PRD §6.5, §7.7, §4 scale).

Longitudinal memory quality must be reported *as a function of depth* — recall@k
as the number of intervening sessions grows — not just at one size. This builds
needle-across-sessions scenarios at increasing depth and reports the recall curve
per backend, alongside storage footprint, so a lab can see how a system degrades
(or holds) as history accumulates.
"""

from __future__ import annotations

from mem_eval.adapters import get_backend
from mem_eval.data.generators import needle
from mem_eval.data.schema import NEEDLE, Suite
from mem_eval.data.suite import DATASET_VERSION, _category_rng
from mem_eval.runner.orchestrate import DEFAULT_K, run_eval

DEFAULT_DEPTHS = (8, 16, 32, 64)


def depth_curve(
    backend_name: str,
    *,
    seed: int = 42,
    depths=DEFAULT_DEPTHS,
    k: int = DEFAULT_K,
    count: int = 2,
) -> list[dict]:
    """recall@k for the needle category at each depth (#sessions). Deterministic."""
    curve = []
    for d in depths:
        scenarios = needle.generate(_category_rng(seed, 4), count=count, needle_depth=d)
        suite = Suite(
            suite="v1", dataset_version=DATASET_VERSION, seed=seed,
            scenarios=scenarios, config_name=f"needle-depth-{d}",
        )
        backend = get_backend(backend_name)
        sc = run_eval(backend, suite, k=k)
        m = sc["metrics"]["by_category"][NEEDLE]
        storage = m["storage"]
        curve.append({
            "depth": d,
            "sessions": d,
            "recall_at_k": m["recall_at_k"],
            "answer_accuracy": m["answer_accuracy"],
            "bytes": list(storage["bytes_at_n"].values())[-1] if storage["bytes_at_n"] else 0,
        })
    return curve


def render_depth_curve(backend_name: str, curve: list[dict]) -> str:
    lines = [f"# depth-scaling: {backend_name}", "", "| depth (sessions) | recall@k | answer_acc | bytes |", "| --- | --- | --- | --- |"]
    for row in curve:
        lines.append(
            f"| {row['depth']} | {row['recall_at_k']:.3f} | {row['answer_accuracy']:.3f} | {row['bytes']} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = ["depth_curve", "render_depth_curve", "DEFAULT_DEPTHS"]

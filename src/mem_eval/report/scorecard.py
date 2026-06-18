"""Scorecard JSON emit, determinism hash, and markdown compare (PRD §10.2/10.3, A6)."""

from __future__ import annotations

import hashlib
import json
import os

from mem_eval.data.schema import CATEGORIES
from mem_eval.metrics import QUALITY_KEYS

_ROUND = 6


def determinism_hash(
    metrics_block: dict,
    *,
    backend: dict,
    dataset: dict,
    config: dict,
) -> str:
    """Hash over quality metrics (1-4) + dataset/backend/config identity.
    Operational metrics (latency/cost/storage) are deliberately excluded so
    identical-quality runs are not flagged by environment jitter (PRD §10.3)."""

    def quality(block: dict) -> dict:
        return {k: round(float(block[k]), _ROUND) for k in QUALITY_KEYS}

    payload = {
        "backend": {"name": backend["name"], "version": backend["version"]},
        "dataset": {
            "suite": dataset["suite"],
            "version": dataset["version"],
            "seed": dataset["seed"],
            "scale": dataset.get("scale", "standard"),
        },
        "config": {"k": config["k"], "consolidate_cadence": config["consolidate_cadence"]},
        "overall": quality(metrics_block["overall"]),
        "by_category": {c: quality(metrics_block["by_category"][c]) for c in CATEGORIES},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _run_id(backend: dict, dataset: dict, config: dict) -> str:
    scale = dataset.get("scale", "standard")
    seed = (
        f"{backend['name']}:{backend['version']}|"
        f"{dataset['suite']}:{dataset['version']}:{dataset['seed']}:{scale}|"
        f"k={config['k']}:{config['consolidate_cadence']}"
    )
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def build_scorecard(
    *,
    backend: dict,
    dataset: dict,
    config: dict,
    metrics_block: dict,
    env: dict,
) -> dict:
    dhash = determinism_hash(metrics_block, backend=backend, dataset=dataset, config=config)
    return {
        "run_id": _run_id(backend, dataset, config),
        "backend": backend,
        "dataset": dataset,
        "config": config,
        "metrics": metrics_block,
        "env": env,
        "determinism_hash": dhash,
    }


def write_scorecard(scorecard: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    b = scorecard["backend"]["name"]
    d = scorecard["dataset"]
    scale = d.get("scale", "standard")
    path = os.path.join(out_dir, f"{b}__{d['suite']}__{scale}__seed{d['seed']}.json")
    with open(path, "w") as fh:
        json.dump(scorecard, fh, indent=2, sort_keys=True)
    return path


# --- A6: compare ------------------------------------------------------------

def _ci_hw(m: dict, key: str) -> float:
    return m.get("ci95", {}).get(key, {}).get("half_width", 0.0)


_COMPARE_ROWS = [
    ("recall@k", lambda m: m["recall_at_k"]),
    ("recall@k ±95%CI", lambda m: _ci_hw(m, "recall_at_k")),
    ("precision@k", lambda m: m["precision_at_k"]),
    ("contradiction_acc", lambda m: m["contradiction_resolution_accuracy"]),
    ("contradiction_acc ±95%CI", lambda m: _ci_hw(m, "contradiction_resolution_accuracy")),
    ("staleness", lambda m: m["staleness"]),
    ("answer_acc", lambda m: m["answer_accuracy"]),
    ("latency_p50_ms (in-proc)", lambda m: m["latency_ms"]["p50"]),
    ("cost_per_query_usd (modeled)", lambda m: m["cost_per_query_usd"]),
    ("storage_growth_slope", lambda m: m["storage"]["growth_slope"]),
]


def compare_scorecards(scorecards: list[dict]) -> str:
    """Render a single side-by-side scoreboard across >=2 backends (PRD §10.1, A6)."""
    if not scorecards:
        return "(no scorecards)\n"
    names = [sc["backend"]["name"] for sc in scorecards]
    overalls = [sc["metrics"]["overall"] for sc in scorecards]

    header = "| metric | " + " | ".join(names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(names)) + " |"
    lines = [
        f"# mem-eval compare ({scorecards[0]['dataset']['suite']}, seed {scorecards[0]['dataset']['seed']})",
        "",
        "_Quality metrics (recall/precision/contradiction/staleness) are measured and "
        "carry CIs. Operational rows are a deterministic MODEL for relative comparison: "
        "cost is priced from token counts, latency is in-process — not infra measurements._",
        "",
        header,
        sep,
    ]
    for label, fn in _COMPARE_ROWS:
        cells = []
        for m in overalls:
            try:
                v = fn(m)
            except (KeyError, TypeError):
                v = float("nan")
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "determinism_hash",
    "build_scorecard",
    "write_scorecard",
    "compare_scorecards",
]

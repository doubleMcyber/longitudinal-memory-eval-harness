"""Real head-to-head on a SHARED capable model served via the local OpenAI-compatible MPS
endpoint (tools/mps_openai_server.py, Qwen3-1.7B fp16+eager). Mem0 drives its LLM through the
endpoint (MEM0_OPENAI_BASE); CB uses its deterministic heuristic (its real zero-LLM-cost config);
temporal_rag is the reference. Subset of category-diverse scenarios so it finishes at ~4.5 tok/s.

Incremental per-scenario saves to results/endpoint_subset.json so a slow/interrupted run still
yields partial results. Run:
  MEM0_OPENAI_BASE=http://127.0.0.1:11435/v1 PYTHONPATH=src python bench_endpoint_subset.py
"""
import copy
import json
import os
import time

from mem_eval.adapters import get_backend
from mem_eval.adapters.mem0_local import Mem0Local
from mem_eval.data.suite import build_suite
from mem_eval.runner.orchestrate import run_eval

OUT = "results/endpoint_subset.json"
KEYS = ["answer_accuracy", "recall_at_k", "precision_at_k",
        "contradiction_resolution_accuracy", "cost_per_query_usd"]
# one scenario per architecture-revealing category (kept small for the slow local endpoint)
WANT_CATS = ["contradiction", "multi_hop", "longitudinal_recall"]


def main():
    assert os.environ.get("MEM0_OPENAI_BASE"), "set MEM0_OPENAI_BASE to the endpoint"
    os.makedirs("results", exist_ok=True)
    suite = build_suite("v1", seed=42, scale="small")
    picked, seen = [], set()
    for sc in suite.scenarios:
        if sc.category in WANT_CATS and sc.category not in seen:
            picked.append(sc)
            seen.add(sc.category)
    print(f"scenarios: {[(s.scenario_id, s.category) for s in picked]}", flush=True)

    mem0 = Mem0Local()
    backends = {"curated_brain": get_backend("curated_brain"),
                "temporal_rag": get_backend("temporal_rag"), "mem0_local": mem0}
    rows = []
    for idx, sc in enumerate(picked):
        sub = copy.copy(suite)
        sub.scenarios = [sc]
        row = {"scenario": sc.scenario_id, "category": sc.category, "nq": len(sc.queries)}
        for name, be in backends.items():
            t = time.time()
            try:
                m = run_eval(be, sub, k=10)["metrics"]["overall"]
                row[name] = {k: round(m[k], 4) for k in KEYS}
                row[name]["secs"] = round(time.time() - t)
            except Exception as e:
                row[name] = {"error": repr(e)[:200], "secs": round(time.time() - t)}
        rows.append(row)
        agg = {}
        for name in backends:
            ok = [r for r in rows if name in r and "error" not in r[name]]
            tot = sum(r["nq"] for r in ok)
            if tot:
                agg[name] = {k: round(sum(r[name][k] * r["nq"] for r in ok) / tot, 4) for k in KEYS}
        with open(OUT, "w") as f:
            json.dump({"done": idx + 1, "total": len(picked), "aggregate": agg,
                       "per_scenario": rows}, f, indent=2)
        print(f"[{idx + 1}/{len(picked)}] {sc.scenario_id} "
              f"mem0={row['mem0_local'].get('secs')}s "
              f"cb_ans={row['curated_brain'].get('answer_accuracy')} "
              f"mem0_ans={row['mem0_local'].get('answer_accuracy')}", flush=True)
    print("SUBSET DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("FAILED:", repr(e), flush=True)

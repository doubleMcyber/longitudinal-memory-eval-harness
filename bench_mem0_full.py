"""Full SMALL-suite offline head-to-head: CB vs temporal_rag vs Mem0 (Qwen-2B), scored by
the harness, run scenario-by-scenario with incremental saves so a mid-run failure still
yields partial results. Mem0 is CPU-bound (~2.7 min/add) -> the whole run is ~5h."""
import copy, json, time

from mem_eval.data.suite import build_suite
from mem_eval.runner.orchestrate import run_eval
from mem_eval.adapters import get_backend
from mem_eval.adapters.mem0_local import Mem0Local

OUT = "/tmp/mem0_full_result.json"
KEYS = ["answer_accuracy", "recall_at_k", "precision_at_k",
        "contradiction_resolution_accuracy", "cost_per_query_usd"]


def main():
    suite = build_suite("v1", seed=42, scale="small")
    mem0 = Mem0Local()  # reused across scenarios (reset() rebuilds store, keeps loaded LLM)
    backends = {"curated_brain": get_backend("curated_brain"),
                "temporal_rag": get_backend("temporal_rag"), "mem0_local": mem0}
    per_scenario = []  # list of {scenario, category, nq, <backend>: {metrics}}

    for idx, sc in enumerate(suite.scenarios):
        sub = copy.copy(suite); sub.scenarios = [sc]
        nq = len(sc.queries)
        row = {"scenario": sc.scenario_id, "category": sc.category, "nq": nq}
        for name, be in backends.items():
            t = time.time()
            try:
                m = run_eval(be, sub, k=10)["metrics"]["overall"]
                row[name] = {k: m[k] for k in KEYS}
                row[name]["secs"] = round(time.time() - t)
            except Exception as e:
                row[name] = {"error": repr(e), "secs": round(time.time() - t)}
        per_scenario.append(row)
        # aggregate so far: query-weighted mean per backend per metric
        agg = {}
        for name in backends:
            tot = sum(r["nq"] for r in per_scenario if name in r and "error" not in r[name])
            if tot:
                agg[name] = {k: round(sum(r[name][k] * r["nq"] for r in per_scenario
                             if name in r and "error" not in r[name]) / tot, 4) for k in KEYS}
                agg[name]["nq"] = tot
        with open(OUT, "w") as f:
            json.dump({"done_scenarios": idx + 1, "total": len(suite.scenarios),
                       "aggregate": agg, "per_scenario": per_scenario}, f, indent=2)
        print(f"[{idx+1}/{len(suite.scenarios)}] {sc.scenario_id} ({sc.category}) "
              f"mem0={row['mem0_local'].get('secs')}s cb_ans={row['curated_brain'].get('answer_accuracy')}",
              flush=True)
    print("FULL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("FULL FAILED:", repr(e), flush=True)

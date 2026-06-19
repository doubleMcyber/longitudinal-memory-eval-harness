"""Offline named-rival head-to-head on a tractable subset: Curated Brain vs Mem0,
both on a cached local model (Qwen-2B) + the same deterministic embedder, scored by the
harness. mem0 is CPU-bound (~2.7 min/add), so we use the smallest scenario per key category."""
import copy, json, sys, time

from mem_eval.data.suite import build_suite
from mem_eval.runner.orchestrate import run_eval
from mem_eval.adapters import get_backend
from mem_eval.adapters.mem0_local import Mem0Local

CATS = ["contradiction", "multi_hop", "longitudinal_recall"]


def main():
    suite = build_suite("v1", seed=42, scale="small")
    # smallest-turn scenario per key category -> a tractable but representative subset
    chosen = []
    for c in CATS:
        scs = [s for s in suite.scenarios if s.category == c]
        scs.sort(key=lambda s: sum(len(ss.turns) for ss in s.sessions))
        if scs:
            chosen.append(scs[0])
    sub = copy.copy(suite)
    sub.scenarios = chosen
    turns = sum(len(ss.turns) for s in chosen for ss in s.sessions)
    qn = sum(len(s.queries) for s in chosen)
    print(f"subset: {[s.scenario_id for s in chosen]} | {turns} turns, {qn} queries "
          f"| est mem0 ingest ~{turns*2.7:.0f} min", flush=True)

    out = {}
    for name, be in [("curated_brain", get_backend("curated_brain")),
                     ("temporal_rag", get_backend("temporal_rag")),
                     ("mem0_local", Mem0Local())]:
        t = time.time()
        try:
            card = run_eval(be, sub, k=10)
            m = card["metrics"]["overall"]
            out[name] = {"answer_acc": m["answer_accuracy"], "recall": m["recall_at_k"],
                         "precision": m["precision_at_k"],
                         "contradiction": m["contradiction_resolution_accuracy"],
                         "cost_per_query": m["cost_per_query_usd"], "secs": round(time.time()-t)}
            print(f"{name}: {out[name]}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[name] = {"error": repr(e), "secs": round(time.time()-t)}
            print(f"{name} FAILED: {e!r}", flush=True)

    with open("/tmp/mem0_h2h_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("H2H DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("H2H FAILED:", repr(e), flush=True)
        sys.exit(1)

"""mem-eval CLI (PRD §10.1).

    mem-eval run     --backend naive_rag --suite v1 --seed 42 --k 10 --out results/
    mem-eval compare results/*.json
    mem-eval gen     --suite v1 --seed 42 --out datasets/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from mem_eval.adapters import get_backend
from mem_eval.data.suite import PRESETS, build_suite
from mem_eval.data.schema import suite_content_hash
from mem_eval.report.scorecard import compare_scorecards, write_scorecard
from mem_eval.runner.analyze import DEFAULT_DEPTHS, depth_curve, render_depth_curve
from mem_eval.runner.orchestrate import DEFAULT_CADENCE, DEFAULT_K, run_eval


def _cmd_run(args) -> int:
    suite = build_suite(args.suite, seed=args.seed, scale=args.scale)
    backend = get_backend(args.backend)
    scorecard = run_eval(
        backend, suite, k=args.k, consolidate_cadence=args.consolidate_cadence
    )
    path = write_scorecard(scorecard, args.out)
    print(f"wrote {path}")
    o = scorecard["metrics"]["overall"]
    print(
        f"  recall@k={o['recall_at_k']:.3f} precision@k={o['precision_at_k']:.3f} "
        f"contradiction_acc={o['contradiction_resolution_accuracy']:.3f} "
        f"staleness={o['staleness']:.3f}"
    )
    print(f"  determinism_hash={scorecard['determinism_hash']}")
    return 0


def _cmd_compare(args) -> int:
    paths: list[str] = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) or [p])
    scorecards = []
    for p in paths:
        with open(p) as fh:
            scorecards.append(json.load(fh))
    sys.stdout.write(compare_scorecards(scorecards))
    return 0


def _cmd_iaa(args) -> int:
    from mem_eval.data.importers.transcript import transcript_candidate_refs
    from mem_eval.grading.agreement import inter_annotator_agreement

    with open(args.annotations) as fh:
        multi = json.load(fh)
    refs = transcript_candidate_refs(args.transcript)
    result = inter_annotator_agreement(multi, refs)
    print(f"support_kappa={result['support_kappa']:.3f} "
          f"answer_agreement={result['answer_agreement']:.3f} "
          f"annotators={result['n_annotators']}")
    for pair, kv in result["pairwise_kappa"].items():
        print(f"  {pair}: kappa={kv:.3f}")
    return 0


def _cmd_depth(args) -> int:
    depths = tuple(int(d) for d in args.depths.split(",")) if args.depths else DEFAULT_DEPTHS
    curve = depth_curve(args.backend, seed=args.seed, depths=depths, k=args.k)
    sys.stdout.write(render_depth_curve(args.backend, curve))
    return 0


def _cmd_gen(args) -> int:
    suite = build_suite(args.suite, seed=args.seed, scale=args.scale)
    os.makedirs(args.out, exist_ok=True)
    payload = {
        "suite": suite.suite,
        "dataset_version": suite.dataset_version,
        "seed": suite.seed,
        "scale": suite.config_name,
        "content_hash": suite_content_hash(suite),
        "num_scenarios": len(suite.scenarios),
        "num_queries": sum(len(s.queries) for s in suite.scenarios),
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "category": s.category,
                "num_sessions": len(s.sessions),
                "num_facts": len(s.facts),
                "queries": [q.query_id for q in s.queries],
            }
            for s in suite.scenarios
        ],
    }
    path = os.path.join(args.out, f"{suite.suite}__{suite.config_name}__seed{suite.seed}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"wrote {path} (content_hash={payload['content_hash'][:16]}…)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mem-eval", description="Longitudinal Memory Eval Harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a backend on a suite and emit a scorecard")
    r.add_argument("--backend", required=True)
    r.add_argument("--suite", default="v1")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--k", type=int, default=DEFAULT_K)
    r.add_argument("--scale", choices=sorted(PRESETS), default=None,
                   help="suite scale preset (default: standard)")
    r.add_argument("--consolidate-cadence", dest="consolidate_cadence", default=DEFAULT_CADENCE)
    r.add_argument("--out", default="results/")
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("compare", help="side-by-side scoreboard across scorecard JSONs")
    c.add_argument("paths", nargs="+")
    c.set_defaults(func=_cmd_compare)

    g = sub.add_parser("gen", help="materialize a dataset")
    g.add_argument("--suite", default="v1")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--scale", choices=sorted(PRESETS), default=None)
    g.add_argument("--out", default="datasets/")
    g.set_defaults(func=_cmd_gen)

    d = sub.add_parser("depth", help="recall@k vs depth (#sessions) curve for a backend")
    d.add_argument("--backend", required=True)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--k", type=int, default=DEFAULT_K)
    d.add_argument("--depths", default=None, help="comma-separated depths, e.g. 8,16,32,64")
    d.set_defaults(func=_cmd_depth)

    a = sub.add_parser("iaa", help="inter-annotator agreement over a multi-annotator sidecar")
    a.add_argument("--transcript", required=True)
    a.add_argument("--annotations", required=True, help="multi-annotator JSON sidecar")
    a.set_defaults(func=_cmd_iaa)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

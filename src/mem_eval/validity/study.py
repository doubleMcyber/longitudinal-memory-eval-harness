"""External-validity study — synthetic vs real-log rank correlation (D3).

The headline question a benchmark must answer before labs trust it: *does ranking
memory systems on our synthetic suite predict how they rank on real conversation
logs?* If the orderings agree, the (cheap, scalable, gold-exact) synthetic suite
is a valid proxy for the (expensive, hand-labeled) real one.

We rank a fixed panel of backends on BOTH suites by a transparent quality
composite, then report Spearman's rho (headline) + Kendall's tau + an exact
permutation p-value, plus a per-axis decomposition so the correlation is
attributable to genuine capability transfer (contradiction handling, paraphrase
robustness, recall) rather than one lucky aggregate.

Composite axes are the four higher-is-better quality metrics that are well
defined and hand-labelable on BOTH suites: recall@k, precision@k, contradiction-
resolution accuracy, and answer accuracy. Staleness is deliberately excluded: it
requires the synthetic temporal fact graph (valid_from/entity/attribute), which a
hand-labeled real corpus does not reconstruct, so it is constant on real logs and
would only add noise to the cross-suite comparison.

Honest scope: the shipped real corpus is a curated, hand-authored stand-in for
human-collected production logs (see corpora/real_logs_v1/README.md). The study
is the *instrument*; swapping in a genuinely human-collected, IAA-vetted corpus
of the same format reuses every line of this module unchanged. The claim is
bounded and conditional: to the extent real logs exercise these capabilities, the
synthetic ranking predicts the real ranking.
"""

from __future__ import annotations

from mem_eval.adapters import get_backend
from mem_eval.data.importers.transcript import DEFAULT_CORPUS, build_suite_from_corpus
from mem_eval.data.suite import build_suite
from mem_eval.metrics.correlation import kendall_tau, permutation_pvalue, spearman_rho
from mem_eval.runner.orchestrate import run_eval

# Higher-is-better quality axes defined + hand-labelable on BOTH suites.
COMPOSITE_AXES = (
    "recall_at_k",
    "precision_at_k",
    "contradiction_resolution_accuracy",
    "answer_accuracy",
)

# The default ranked panel: floor -> naive ceiling -> curated/semantic references.
DEFAULT_PANEL = ("no_memory", "long_context", "naive_rag", "temporal_rag", "semantic_rag")

_FIXED_TS = "1970-01-01T00:00:00"  # metrics are timestamp-independent; pin for reproducibility


def composite_score(overall: dict) -> float:
    """Equal-weight mean of the composite axes (all in [0,1], higher better)."""
    return sum(overall[a] for a in COMPOSITE_AXES) / len(COMPOSITE_AXES)


def score_panel(suite, panel=DEFAULT_PANEL, *, k: int = 10) -> dict:
    """Run each backend on ``suite`` and return its composite + per-axis scores."""
    scores: dict[str, dict] = {}
    for name in panel:
        overall = run_eval(get_backend(name), suite, k=k, timestamp=_FIXED_TS)["metrics"]["overall"]
        scores[name] = {"composite": composite_score(overall),
                        **{a: overall[a] for a in COMPOSITE_AXES}}
    return scores


def _ranking(scores: dict) -> list[str]:
    """Backend names best -> worst by composite (stable tie-break on name)."""
    return sorted(scores, key=lambda n: (-scores[n]["composite"], n))


def external_validity(synthetic, real, panel=DEFAULT_PANEL, *, k: int = 10) -> dict:
    """Rank ``panel`` on both suites; report rank correlation + per-axis decomposition."""
    panel = tuple(panel)
    syn = score_panel(synthetic, panel, k=k)
    rl = score_panel(real, panel, k=k)

    syn_comp = [syn[n]["composite"] for n in panel]
    rl_comp = [rl[n]["composite"] for n in panel]

    rho = spearman_rho(syn_comp, rl_comp)
    per_axis = {
        a: spearman_rho([syn[n][a] for n in panel], [rl[n][a] for n in panel])
        for a in COMPOSITE_AXES
    }
    syn_rank, rl_rank = _ranking(syn), _ranking(rl)
    return {
        "panel": list(panel),
        "k": k,
        "composite_axes": list(COMPOSITE_AXES),
        "synthetic": {"suite": synthetic.suite, "seed": synthetic.seed,
                      "scale": synthetic.config_name, "scores": syn, "ranking": syn_rank},
        "real": {"suite": real.suite, "scores": rl, "ranking": rl_rank},
        "spearman_rho": rho,
        "kendall_tau": kendall_tau(syn_comp, rl_comp),
        "p_value": permutation_pvalue(syn_comp, rl_comp),
        "per_axis_spearman": per_axis,
        "rankings_identical": syn_rank == rl_rank,
        "winner_synthetic": syn_rank[0],
        "winner_real": rl_rank[0],
        "floor_last_on_both": syn_rank[-1] == rl_rank[-1] == "no_memory",
    }


def run_default_study(*, seed: int = 42, scale: str | None = None,
                      corpus_path: str = DEFAULT_CORPUS, k: int = 10,
                      panel=DEFAULT_PANEL) -> dict:
    """Convenience: synthetic v1 (seed/scale) vs the shipped real corpus."""
    synthetic = build_suite("v1", seed=seed, scale=scale)
    real = build_suite_from_corpus(corpus_path)
    return external_validity(synthetic, real, panel, k=k)


def _interpret(rho: float) -> str:
    a = abs(rho)
    strength = ("negligible" if a < 0.1 else "weak" if a < 0.4 else "moderate"
                if a < 0.7 else "strong" if a < 0.9 else "very strong")
    sign = "positive" if rho >= 0 else "NEGATIVE (orderings disagree)"
    return f"{strength} {sign}"


def render_validity_report(report: dict) -> str:
    panel = report["panel"]
    syn, rl = report["synthetic"]["scores"], report["real"]["scores"]
    syn_rank = {n: i + 1 for i, n in enumerate(report["synthetic"]["ranking"])}
    rl_rank = {n: i + 1 for i, n in enumerate(report["real"]["ranking"])}

    lines = [
        "# External-validity study — synthetic vs real-log ranking (D3)",
        "",
        f"Panel of {len(panel)} backends ranked by the quality composite "
        f"(mean of {', '.join(report['composite_axes'])}) on the synthetic suite "
        f"(`{report['synthetic']['suite']}`, seed {report['synthetic']['seed']}, "
        f"scale {report['synthetic']['scale']}) and the real-log corpus "
        f"(`{report['real']['suite']}`).",
        "",
        "| backend | synth composite | synth rank | real composite | real rank |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for n in sorted(panel, key=lambda m: syn_rank[m]):
        lines.append(
            f"| {n} | {syn[n]['composite']:.4f} | {syn_rank[n]} "
            f"| {rl[n]['composite']:.4f} | {rl_rank[n]} |"
        )

    lines += [
        "",
        f"**Spearman ρ = {report['spearman_rho']:.4f}** ({_interpret(report['spearman_rho'])}) "
        f"· Kendall τ = {report['kendall_tau']:.4f} "
        f"· exact permutation p = {report['p_value']:.4f}",
        "",
        f"Rankings identical: {report['rankings_identical']} · "
        f"synthetic winner: {report['winner_synthetic']} · real winner: {report['winner_real']} · "
        f"do-nothing floor last on both: {report['floor_last_on_both']}",
        "",
        "Per-axis Spearman (where does the agreement come from?):",
        "",
        "| axis | ρ |",
        "| --- | ---: |",
    ]
    for axis, rho in report["per_axis_spearman"].items():
        lines.append(f"| {axis} | {rho:.4f} |")
    lines += [
        "",
        "_The real corpus is a curated, hand-authored stand-in for human-collected "
        "production logs; gold is hand-labeled (IAA-vetted), never inferred. The claim "
        "is bounded: to the extent real logs exercise these capabilities, the synthetic "
        "ranking predicts the real one. Staleness is excluded from the composite (it "
        "needs the synthetic temporal fact graph)._",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "COMPOSITE_AXES",
    "DEFAULT_PANEL",
    "composite_score",
    "score_panel",
    "external_validity",
    "run_default_study",
    "render_validity_report",
]

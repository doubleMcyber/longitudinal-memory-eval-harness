"""Acceptance suite encoding PRD §11 (A1–A7) — the gate.

`pytest -q` exiting 0 with this suite present is the single command that proves
the build is healthy (PRD §11, §13).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mem_eval.adapters import BASELINES, get_backend
from mem_eval.adapters.base import QueryResult
from mem_eval.data.schema import (
    CATEGORIES,
    CONTRADICTION,
    MULTI_HOP,
    scenario_content_hash,
    suite_content_hash,
)
from mem_eval.data.suite import build_suite
from mem_eval.report.scorecard import compare_scorecards
from mem_eval.runner.orchestrate import run_eval

SEED = 42
K = 10

# 7 metric keys that must appear in every metrics block (PRD §7, A3).
SEVEN_METRIC_KEYS = (
    "recall_at_k",
    "precision_at_k",
    "contradiction_resolution_accuracy",
    "staleness",
    "latency_ms",
    "cost_per_query_usd",
    "storage",
)


@pytest.fixture(scope="module")
def suite():
    return build_suite("v1", seed=SEED)


def _run(name: str, suite, **kw) -> dict:
    return run_eval(get_backend(name), suite, k=K, **kw)


# --- A1 — Contract + baselines run -----------------------------------------


def test_a1_three_baselines_run_end_to_end(suite):
    assert set(BASELINES) == {"no_memory", "long_context", "naive_rag"}
    for name in BASELINES:
        sc = _run(name, suite)
        assert sc["backend"]["name"] == name
        assert sc["metrics"]["overall"]["num_queries"] > 0


def test_a1_query_returns_well_formed_result(suite):
    scenario = suite.scenarios[0]
    q = scenario.queries[0]
    for name in BASELINES:
        b = get_backend(name)
        b.reset()
        for s in scenario.sessions:
            b.ingest(s)
        res = b.query(q.prompt, K, q.as_of)
        assert isinstance(res, QueryResult)
        assert isinstance(res.items, list)
        assert res.usage is not None
        assert res.latency_ms >= 0.0


# --- A2 — Coverage with exact labels, deterministic -------------------------


def test_a2_all_five_categories_have_generators(suite):
    for cat in CATEGORIES:
        scs = suite.scenarios_for(cat)
        assert scs, f"no scenarios for category {cat}"
        assert any(q.gold_support for s in scs for q in s.queries), cat
    assert len(CATEGORIES) == 5


def test_a2_gold_labels_are_exact(suite):
    """Gold answers come from the structured layer, not inferred from text, and
    are pinned to the CORRECT supporting fact (not merely *some* support value)."""
    for scenario in suite.scenarios:
        by_id = scenario.fact_by_id()
        for q in scenario.queries:
            assert q.gold_answer, q.query_id
            support_vals = [by_id[f].value for f in q.gold_support if f in by_id]
            assert support_vals, f"{q.query_id}: synthetic query has no structured support"
            if q.category == MULTI_HOP:
                # the answer must be the TERMINAL hop's value, not just any hop —
                # otherwise a wrong-hop label would pass (the loose-membership trap).
                terminal = by_id[q.gold_support[-1]].value
                assert q.gold_answer == terminal, q.query_id
            else:
                assert q.gold_answer in support_vals, q.query_id


def test_a2_determinism_same_seed_same_hash():
    h1 = suite_content_hash(build_suite("v1", seed=SEED))
    h2 = suite_content_hash(build_suite("v1", seed=SEED))
    assert h1 == h2
    # different seed should (overwhelmingly) change the dataset
    h3 = suite_content_hash(build_suite("v1", seed=SEED + 1))
    assert h1 != h3


def test_a2_per_scenario_hash_stable():
    s1 = build_suite("v1", seed=SEED).scenarios
    s2 = build_suite("v1", seed=SEED).scenarios
    assert [scenario_content_hash(s) for s in s1] == [scenario_content_hash(s) for s in s2]


# --- A3 — All 7 metrics emitted, per-category and overall -------------------


def test_a3_all_seven_metrics_emitted(suite):
    sc = _run("naive_rag", suite)
    overall = sc["metrics"]["overall"]
    for key in SEVEN_METRIC_KEYS:
        assert key in overall, f"missing overall metric {key}"
    assert set(sc["metrics"]["by_category"]) == set(CATEGORIES)
    for cat in CATEGORIES:
        block = sc["metrics"]["by_category"][cat]
        for key in SEVEN_METRIC_KEYS:
            assert key in block, f"missing {key} in category {cat}"
    # operational metrics carry their detailed shape
    assert set(overall["latency_ms"]) >= {"p50", "p95", "p99"}
    assert set(overall["storage"]) >= {"bytes_at_n", "growth_slope"}


# --- A4 — Discrimination (the crux) ----------------------------------------


def test_a4_discrimination(suite):
    nomem = _run("no_memory", suite)
    naive = _run("naive_rag", suite)

    r_nomem = nomem["metrics"]["overall"]["recall_at_k"]
    r_naive = naive["metrics"]["overall"]["recall_at_k"]
    assert r_nomem < r_naive, f"recall did not discriminate: {r_nomem} !< {r_naive}"

    stale_naive = naive["metrics"]["by_category"][CONTRADICTION]["staleness"]
    assert stale_naive > 0.0, "NaiveRAG must exhibit staleness>0 on the contradiction suite"

    ca_nomem = nomem["metrics"]["overall"]["contradiction_resolution_accuracy"]
    ca_naive = naive["metrics"]["overall"]["contradiction_resolution_accuracy"]
    assert ca_nomem == pytest.approx(0.0, abs=1e-9)
    assert ca_nomem < ca_naive


def test_a4_kband_invariant_holds():
    """The easy difficulty anchor relies on enough same-topic distractors to push
    the stale fact below the top-k cutoff across the tested band (k<=20), so
    NaiveRAG passes at least one contradiction scenario for every seed. If a
    future edit lowers MAX_DISTRACTORS below that, ca(NaiveRAG) can collapse to
    the NoMemory floor and A4 silently breaks. Pin the invariant."""
    from mem_eval.data.generators.contradiction import MAX_DISTRACTORS

    assert MAX_DISTRACTORS >= 20, (
        "MAX_DISTRACTORS must keep the easy anchor's stale fact below the cutoff "
        "for the k-band the robustness test exercises (up to k=20)"
    )


@pytest.mark.parametrize("seed", [42, 7, 123])
@pytest.mark.parametrize("k", [3, 5, 10, 20])
def test_a4_discrimination_is_robust_across_k_and_seed(seed, k):
    """A4 must not be tuned to one k: the separation holds across a band of k
    and multiple seeds, not just the default."""
    s = build_suite("v1", seed=seed)
    nomem = run_eval(get_backend("no_memory"), s, k=k)["metrics"]
    naive = run_eval(get_backend("naive_rag"), s, k=k)["metrics"]
    assert nomem["overall"]["recall_at_k"] < naive["overall"]["recall_at_k"]
    assert naive["by_category"][CONTRADICTION]["staleness"] > 0.0
    assert nomem["overall"]["contradiction_resolution_accuracy"] == 0.0
    assert naive["overall"]["contradiction_resolution_accuracy"] > 0.0


@pytest.mark.parametrize("seed", [42, 7, 123, 2024])
def test_meaningful_spectrum_reference_beats_naive(seed):
    """A legit benchmark must REWARD good behavior, not only punish nothing: the
    contradiction-aware reference (TemporalRAG) strictly beats NaiveRAG on the
    contradiction axis, and the full floor->naive->good ordering holds. This is
    the property that lets a real memory system see a credible target to clear."""
    s = build_suite("v1", seed=seed)
    nomem = run_eval(get_backend("no_memory"), s, k=10)["metrics"]
    naive = run_eval(get_backend("naive_rag"), s, k=10)["metrics"]
    temporal = run_eval(get_backend("temporal_rag"), s, k=10)["metrics"]

    # contradiction: floor < naive < good reference, with HEADROOM above the
    # reference (the benchmark is not tuned to let its own reference score 1.0).
    ca_nm = nomem["overall"]["contradiction_resolution_accuracy"]
    ca_nr = naive["overall"]["contradiction_resolution_accuracy"]
    ca_tr = temporal["overall"]["contradiction_resolution_accuracy"]
    assert ca_nm < ca_nr < ca_tr < 1.0, (ca_nm, ca_nr, ca_tr)

    # staleness: the good reference curates away stale items the naive one returns
    assert (
        temporal["by_category"][CONTRADICTION]["staleness"]
        < naive["by_category"][CONTRADICTION]["staleness"]
    )

    # retrieval quality: reference is at least as good and strictly beats the floor
    assert nomem["overall"]["recall_at_k"] < naive["overall"]["recall_at_k"]
    assert naive["overall"]["recall_at_k"] <= temporal["overall"]["recall_at_k"]


def test_reference_has_emergent_headroom():
    """The contradiction-aware reference must NOT score a flat 1.0 — that would
    look like the benchmark is co-designed to let its own reference win. Adversarial
    coreference phrasing it cannot resolve leaves headroom, so its accuracy is < 1.0
    and varies across seeds (emergent), leaving room for a real system to claim."""
    vals = [
        run_eval(get_backend("temporal_rag"), build_suite("v1", seed=s), k=10)["metrics"][
            "overall"
        ]["contradiction_resolution_accuracy"]
        for s in range(40, 52)
    ]
    assert all(v < 1.0 for v in vals), f"reference scored a suspicious 1.0: {vals}"
    assert len(set(vals)) >= 3, f"reference score looks templated, not emergent: {vals}"


@pytest.mark.parametrize("seed", [42, 7, 123])
def test_multihop_exposes_assembly_gap_with_headroom(seed):
    """Multi-hop chains (2-3 hops, RNG length) surface the retrieval-vs-assembly
    gap: the in-box backends retrieve the hops (recall > 0) but cannot assemble the
    terminal answer (answer accuracy ~ 0) — honest headroom for a reasoning system."""
    s = build_suite("v1", seed=seed)
    for name in ("naive_rag", "temporal_rag"):
        mh = run_eval(get_backend(name), s, k=10)["metrics"]["by_category"][MULTI_HOP]
        assert mh["recall_at_k"] > 0.5, (name, mh["recall_at_k"])
        assert mh["answer_accuracy"] < 0.5, (name, mh["answer_accuracy"])


def test_multihop_chain_length_actually_varies():
    """Chains must genuinely span 2 and 3 hops across the seed population — a
    regression collapsing them to one fixed length must fail here."""
    lengths = {
        len(q.gold_support)
        for seed in range(40, 60)
        for sc in build_suite("v1", seed=seed).scenarios_for(MULTI_HOP)
        for q in sc.queries
    }
    assert lengths == {2, 3}, lengths


def test_discrimination_is_emergent_not_constant():
    """Discrimination must be a property of data+mechanism, not a designed
    constant: NaiveRAG's contradiction accuracy varies across seeds (it used to
    be hardcoded to exactly 0.5 by an i%2 split)."""
    vals = {
        run_eval(get_backend("naive_rag"), build_suite("v1", seed=s), k=10)["metrics"][
            "overall"
        ]["contradiction_resolution_accuracy"]
        for s in range(40, 52)
    }
    assert len(vals) >= 3, f"contradiction accuracy looks templated, not emergent: {vals}"


# --- A5 — Determinism -------------------------------------------------------


def test_a5_identical_runs_same_determinism_hash(suite):
    h1 = _run("naive_rag", suite)["determinism_hash"]
    h2 = _run("naive_rag", suite)["determinism_hash"]
    assert h1 == h2


def test_a5_quality_excludes_operational_jitter(suite):
    """Determinism hash must ignore latency/cost/storage (PRD §10.3)."""
    a = _run("naive_rag", suite)
    b = _run("naive_rag", suite)
    # latency will differ run-to-run, but the determinism hash must not.
    assert a["determinism_hash"] == b["determinism_hash"]


# --- A6 — Comparison --------------------------------------------------------


def test_a6_compare_emits_scoreboard(suite):
    cards = [_run("no_memory", suite), _run("naive_rag", suite)]
    md = compare_scorecards(cards)
    lines = md.splitlines()
    header = next(ln for ln in lines if ln.startswith("| metric |"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    # one column per backend (plus the leading 'metric' label column) — a single
    # table across >=2 backends, not a degenerate one-column render.
    assert cols == ["metric", "no_memory", "naive_rag"]
    # at least one metric row must show DIFFERENT values per backend (real data).
    recall_row = next(ln for ln in lines if ln.startswith("| recall@k |"))
    vals = [c.strip() for c in recall_row.strip("|").split("|")[1:]]
    assert vals[0] != vals[1], f"compare did not emit distinct per-backend values: {vals}"


# --- A7 — Gate (meta) -------------------------------------------------------


def test_a7_gate_shape(suite):
    assert len(CATEGORIES) == 5
    assert len(BASELINES) == 3
    assert len(SEVEN_METRIC_KEYS) == 7


def test_reproducibility_manifest_present(suite):
    """Every scorecard must carry a reproducibility manifest (PRD §10.3): backend
    + dataset (incl. scale) + config + env (harness/python/platform/judge/models)
    + determinism hash, so a run can be re-derived and fairly compared."""
    sc = _run("naive_rag", suite)
    assert set(sc) >= {"run_id", "backend", "dataset", "config", "metrics", "env", "determinism_hash"}
    assert {"name", "version"} <= set(sc["backend"])
    assert {"suite", "version", "seed", "scale"} <= set(sc["dataset"])
    env = sc["env"]
    assert {"harness_version", "python", "platform", "embedding_model", "answer_judge"} <= set(env)
    assert sc["determinism_hash"]

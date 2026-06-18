"""Bootstrap confidence interval tests (PRD §10 rigor) — Wave 3."""

from __future__ import annotations

from mem_eval.adapters import get_backend
from mem_eval.data.suite import build_suite
from mem_eval.metrics.significance import bootstrap_ci, difference_ci
from mem_eval.runner.orchestrate import run_eval

QUALITY = ("recall_at_k", "precision_at_k", "contradiction_resolution_accuracy", "staleness")


def test_bootstrap_ci_deterministic_and_brackets_point():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(vals)
    b = bootstrap_ci(vals)
    assert a == b  # deterministic
    assert a["point"] == sum(vals) / len(vals)
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["n"] == len(vals)


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([])["n"] == 0
    one = bootstrap_ci([0.7])
    assert one["lo"] == one["point"] == one["hi"] == 0.7
    # zero-variance sample -> zero-width interval
    z = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert z["lo"] == z["hi"] == 1.0


def test_difference_ci_detects_real_gap():
    low = [0.0, 0.1, 0.0, 0.1, 0.0]
    high = [0.9, 1.0, 0.9, 1.0, 0.9]
    d = difference_ci(low, high)
    assert d["delta"] > 0
    assert d["significant"] is True
    # identical distributions -> not significant
    same = difference_ci(low, list(low))
    assert same["significant"] is False


def test_scorecard_exposes_quality_cis():
    s = build_suite("v1", seed=42, scale="small")
    overall = run_eval(get_backend("naive_rag"), s, k=10)["metrics"]["overall"]
    assert "ci95" in overall
    for key in QUALITY:
        ci = overall["ci95"][key]
        assert set(ci) >= {"point", "lo", "hi", "half_width", "n"}
        assert ci["lo"] <= ci["point"] <= ci["hi"]
    # the point in ci95 matches the reported scalar metric
    assert abs(overall["ci95"]["recall_at_k"]["point"] - overall["recall_at_k"]) < 1e-9

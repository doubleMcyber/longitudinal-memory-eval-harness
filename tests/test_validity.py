"""External-validity study tests (D3): synthetic vs real-log rank correlation."""

from __future__ import annotations

from mem_eval.validity import (
    COMPOSITE_AXES,
    DEFAULT_PANEL,
    composite_score,
    render_validity_report,
    run_default_study,
    score_panel,
)
from mem_eval.data.suite import build_suite


def test_composite_score_is_axis_mean():
    overall = {a: v for a, v in zip(COMPOSITE_AXES, (1.0, 0.0, 0.5, 0.25))}
    assert composite_score(overall) == (1.0 + 0.0 + 0.5 + 0.25) / 4


def test_score_panel_returns_composite_and_axes():
    scores = score_panel(build_suite("v1", seed=42), k=10)
    assert set(scores) == set(DEFAULT_PANEL)
    for s in scores.values():
        assert "composite" in s
        for a in COMPOSITE_AXES:
            assert a in s
            assert 0.0 <= s[a] <= 1.0


def test_study_is_deterministic():
    a = run_default_study()
    b = run_default_study()
    assert a["spearman_rho"] == b["spearman_rho"]
    assert a["kendall_tau"] == b["kendall_tau"]
    assert a["synthetic"]["ranking"] == b["synthetic"]["ranking"]
    assert a["real"]["ranking"] == b["real"]["ranking"]


def test_synthetic_ranking_predicts_real_ranking():
    r = run_default_study()
    # rank correlation is strong and positive (the external-validity claim)
    assert r["spearman_rho"] >= 0.7
    assert r["kendall_tau"] >= 0.6
    # significant for the 5-backend panel
    assert 0.0 < r["p_value"] <= 0.05


def test_winner_and_floor_are_stable_across_suites():
    r = run_default_study()
    # the do-nothing backend is worst on BOTH suites; the same system wins both
    assert r["floor_last_on_both"] is True
    assert r["winner_synthetic"] == r["winner_real"]
    assert r["synthetic"]["scores"]["no_memory"]["composite"] == 0.0
    assert r["real"]["scores"]["no_memory"]["composite"] == 0.0


def test_agreement_is_genuine_capability_transfer_not_one_aggregate():
    r = run_default_study()
    # every composite axis correlates positively across suites -> the agreement is
    # attributable to real capability transfer, not a single lucky metric
    for axis, rho in r["per_axis_spearman"].items():
        assert rho > 0.0, f"axis {axis} did not transfer (rho={rho})"


def test_rankings_identical_on_shipped_data():
    # the shipped synthetic suite + corpus reproduce the SAME ordering end-to-end
    r = run_default_study()
    assert r["rankings_identical"] is True
    assert r["synthetic"]["ranking"][0] == "temporal_rag"
    assert r["synthetic"]["ranking"][-1] == "no_memory"


def test_report_renders_key_facts():
    out = render_validity_report(run_default_study())
    assert "External-validity study" in out
    assert "Spearman" in out
    assert "Per-axis" in out
    for n in DEFAULT_PANEL:
        assert n in out

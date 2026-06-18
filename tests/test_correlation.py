"""Rank-correlation primitive tests (D3 external-validity study)."""

from __future__ import annotations

import math

from mem_eval.metrics.correlation import (
    average_ranks,
    kendall_tau,
    permutation_pvalue,
    spearman_rho,
)


def test_average_ranks_with_ties():
    # values: 10, 20, 20, 40 -> ranks 1, 2.5, 2.5, 4
    assert average_ranks([10, 20, 20, 40]) == [1.0, 2.5, 2.5, 4.0]
    # already-sorted distinct -> 1..n
    assert average_ranks([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_spearman_perfect_and_reversed():
    x = [1, 2, 3, 4, 5]
    assert spearman_rho(x, [10, 20, 30, 40, 50]) == 1.0
    assert spearman_rho(x, [50, 40, 30, 20, 10]) == -1.0


def test_spearman_known_value():
    # textbook: x and y monotonic except one swap -> rho between 0 and 1
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 5, 4]
    rho = spearman_rho(x, y)
    # Spearman with one adjacent swap of ranks 4,5: d^2 sum = 2 -> rho = 1 - 6*2/(5*24)=0.9
    assert math.isclose(rho, 0.9, abs_tol=1e-9)


def test_spearman_constant_is_zero():
    assert spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_kendall_perfect_and_reversed():
    x = [1, 2, 3, 4]
    assert kendall_tau(x, [5, 6, 7, 8]) == 1.0
    assert kendall_tau(x, [8, 7, 6, 5]) == -1.0


def test_kendall_tie_robust():
    # one tie in y should not push |tau| above 1
    tau = kendall_tau([1, 2, 3, 4], [1, 2, 2, 4])
    assert -1.0 <= tau <= 1.0
    assert tau > 0.0


def test_permutation_pvalue_perfect_correlation_is_significant():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    # exactly 2 of 120 permutations reach |rho| = 1: the identity (rho=+1) and the
    # full reversal (rho=-1). Two-sided exact p for a perfect monotone relation on
    # n=5 is therefore 2/120.
    p = permutation_pvalue(x, y)
    assert math.isclose(p, 2 / 120, abs_tol=1e-9)


def test_permutation_pvalue_is_deterministic():
    x = [0.0, 0.39, 0.49, 0.55, 0.75]
    y = [0.1, 0.30, 0.52, 0.60, 0.70]
    assert permutation_pvalue(x, y) == permutation_pvalue(x, y)
    assert 0.0 <= permutation_pvalue(x, y) <= 1.0

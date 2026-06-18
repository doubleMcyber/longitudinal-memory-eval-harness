"""Rank-correlation primitives for the external-validity study (D3).

External validity asks: does ranking memory systems on the *synthetic* suite
predict how they rank on *real* conversation logs? That is a question about
agreement between two orderings, which is exactly what rank correlation measures.

We compute Spearman's rho (the headline, as the roadmap names it) and Kendall's
tau-b as a tie-robust companion, plus an EXACT permutation p-value so a small
panel (5-6 backends) still yields a defensible significance statement. Everything
is pure stdlib and deterministic (the permutation null enumerates all orderings;
no RNG), matching the rest of the harness.
"""

from __future__ import annotations

import math
from itertools import permutations


def average_ranks(values: list[float]) -> list[float]:
    """Fractional (tie-averaged) ranks, 1-based. Ties share the mean of the ranks
    they would occupy, so Spearman degrades gracefully when scores tie."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (0-based) -> 1-based ranks i+1..j+1, averaged
        avg = (i + j) / 2.0 + 1.0
        for p in range(i, j + 1):
            ranks[order[p]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        # at least one ranking is constant -> correlation undefined; report 0
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation = Pearson correlation of the tie-averaged ranks.
    +1 identical ordering, -1 reversed, 0 no monotonic association."""
    if len(x) != len(y):
        raise ValueError("inputs must be equal length")
    if len(x) < 2:
        return 0.0
    return _pearson(average_ranks(x), average_ranks(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b (tie-adjusted). Counts concordant vs discordant pairs,
    normalized by the tie-corrected pair counts so ties don't inflate |tau|."""
    if len(x) != len(y):
        raise ValueError("inputs must be equal length")
    n = len(x)
    if n < 2:
        return 0.0
    concordant = discordant = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def permutation_pvalue(x: list[float], y: list[float], *, statistic=spearman_rho) -> float:
    """Two-sided exact permutation p-value for the observed rank correlation.

    Enumerates every permutation of ``y`` against fixed ``x`` (the exact null of
    "no association") and returns the fraction whose statistic is at least as
    extreme (|stat| >= |observed|) as observed. Deterministic and exact for the
    small panels this study uses; raises if the panel is too large to enumerate."""
    n = len(x)
    if n < 2:
        return 1.0
    if n > 8:
        raise ValueError(f"exact permutation p-value enumerates n! orderings; n={n} too large")
    observed = abs(statistic(x, y))
    total = at_least = 0
    for perm in permutations(y):
        total += 1
        if abs(statistic(x, list(perm))) >= observed - 1e-12:
            at_least += 1
    return at_least / total


__all__ = ["average_ranks", "spearman_rho", "kendall_tau", "permutation_pvalue"]

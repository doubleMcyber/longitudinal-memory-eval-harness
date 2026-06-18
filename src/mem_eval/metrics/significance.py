"""Statistical rigor — bootstrap confidence intervals (PRD §8/§10 spirit).

A score with no uncertainty is not trustworthy: labs need to know whether a delta
between two systems is real or noise. We compute percentile bootstrap confidence
intervals over the per-query metric values, deterministically (fixed resampling
seed) so the intervals are reproducible.

These intervals are REPORTED (surfaced in the scorecard and compare view) but, like
the operational metrics, are excluded from the determinism hash — the hash pins
the point estimates of the four quality metrics (PRD §10.3); the CIs are a derived
diagnostic.
"""

from __future__ import annotations

from random import Random


def bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> dict:
    """Percentile bootstrap CI for the mean of ``values``. Deterministic for a
    fixed seed. Returns point estimate, lo/hi bounds, half-width, and n."""
    n = len(values)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "half_width": 0.0, "n": 0}
    point = sum(values) / n
    if n == 1:
        return {"point": point, "lo": point, "hi": point, "half_width": 0.0, "n": 1}

    rng = Random(seed)
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * n_resamples) - 1)
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)
    lo, hi = means[lo_idx], means[hi_idx]
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "half_width": (hi - lo) / 2.0,
        "n": n,
    }


def difference_ci(
    a_values: list[float],
    b_values: list[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> dict:
    """Bootstrap CI for the difference of means (b - a), unpaired. If the CI
    excludes 0, the difference is significant at the given alpha. Deterministic."""
    na, nb = len(a_values), len(b_values)
    if na == 0 or nb == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "significant": False}
    delta = (sum(b_values) / nb) - (sum(a_values) / na)
    rng = Random(seed)
    diffs = []
    for _ in range(n_resamples):
        sa = sum(a_values[rng.randrange(na)] for _ in range(na)) / na
        sb = sum(b_values[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(sb - sa)
    diffs.sort()
    lo = diffs[max(0, int((alpha / 2) * n_resamples) - 1)]
    hi = diffs[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)]
    return {"delta": delta, "lo": lo, "hi": hi, "significant": (lo > 0 or hi < 0)}


__all__ = ["bootstrap_ci", "difference_ci"]

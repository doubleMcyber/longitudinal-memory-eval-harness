"""Suite assembly (PRD §8 versioning, §10.1 `gen`).

`build_suite(seed)` deterministically materializes the v1 suite: all 5 category
generators with exact gold labels. Same seed => identical scenarios + labels
(asserted via content hash, PRD §8 determinism / A2 / A5).
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators import (
    contradiction,
    longitudinal,
    multi_hop,
    needle,
    recency_relevance,
)
from mem_eval.data.schema import (
    CONTRADICTION,
    LONGITUDINAL_RECALL,
    MULTI_HOP,
    NEEDLE,
    RECENCY_RELEVANCE,
    Suite,
)

DATASET_VERSION = "2026.6"

# Ordered (category, generator) — order is part of the dataset identity.
GENERATORS = [
    (LONGITUDINAL_RECALL, longitudinal.generate),
    (CONTRADICTION, contradiction.generate),
    (MULTI_HOP, multi_hop.generate),
    (RECENCY_RELEVANCE, recency_relevance.generate),
    (NEEDLE, needle.generate),
]


def _category_rng(seed: int, idx: int) -> Random:
    # independent, deterministic stream per category
    return Random((seed & 0x7FFFFFFF) * 1000003 ^ ((idx + 1) * 2654435761 & 0x7FFFFFFF))


def build_suite(suite: str = "v1", seed: int = 42) -> Suite:
    if suite != "v1":
        raise ValueError(f"only the 'v1' suite is defined in this build (got {suite!r})")
    scenarios = []
    for idx, (_cat, gen) in enumerate(GENERATORS):
        scenarios.extend(gen(_category_rng(seed, idx)))
    return Suite(suite=suite, dataset_version=DATASET_VERSION, seed=seed, scenarios=scenarios)


__all__ = ["build_suite", "DATASET_VERSION", "GENERATORS"]

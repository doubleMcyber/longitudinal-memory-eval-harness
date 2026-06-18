"""Suite assembly + scale configuration (PRD §8 versioning, §10.1 `gen`, §4 scale).

`build_suite(seed)` deterministically materializes the v1 suite: all 5 category
generators with exact gold labels. Same seed (+ same scale) => identical
scenarios + labels (asserted via content hash, PRD §8 / A2 / A5).

Scale is configurable via `SuiteConfig` / named presets so labs can run small
(smoke), standard (default), or large/deep (stress) suites. The `standard`
preset reproduces the historical data byte-for-byte, so default runs stay
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from mem_eval.data.generators import (
    contradiction,
    lexical_gap,
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

DATASET_VERSION = "2026.10"


@dataclass(frozen=True)
class SuiteConfig:
    """Knobs controlling suite size/depth."""

    name: str = "standard"
    longitudinal_count: int = 3
    longitudinal_sessions: int = 12
    lexical_gap_count: int = 3
    contradiction_count: int = 10
    multi_hop_count: int = 3
    recency_count: int = 2
    needle_count: int = 2
    needle_depth: int = 24


PRESETS: dict[str, SuiteConfig] = {
    "small": SuiteConfig(
        name="small", longitudinal_count=2, longitudinal_sessions=6, lexical_gap_count=1,
        contradiction_count=4, multi_hop_count=2, recency_count=1,
        needle_count=1, needle_depth=10,
    ),
    "standard": SuiteConfig(),
    "large": SuiteConfig(
        name="large", longitudinal_count=6, longitudinal_sessions=40, lexical_gap_count=6,
        contradiction_count=20, multi_hop_count=6, recency_count=4,
        needle_count=4, needle_depth=80,
    ),
}


def _category_rng(seed: int, idx: int) -> Random:
    # independent, deterministic stream per category
    return Random((seed & 0x7FFFFFFF) * 1000003 ^ ((idx + 1) * 2654435761 & 0x7FFFFFFF))


def resolve_config(scale: str | None = None, config: SuiteConfig | None = None) -> SuiteConfig:
    if config is not None:
        return config
    if scale is not None:
        if scale not in PRESETS:
            raise ValueError(f"unknown scale {scale!r}; choose from {sorted(PRESETS)}")
        return PRESETS[scale]
    return PRESETS["standard"]


def build_suite(
    suite: str = "v1",
    seed: int = 42,
    *,
    scale: str | None = None,
    config: SuiteConfig | None = None,
) -> Suite:
    if suite != "v1":
        raise ValueError(f"only the 'v1' suite is defined in this build (got {suite!r})")
    cfg = resolve_config(scale, config)
    scenarios = []
    scenarios += longitudinal.generate(
        _category_rng(seed, 0), count=cfg.longitudinal_count, num_sessions=cfg.longitudinal_sessions
    )
    scenarios += lexical_gap.generate(_category_rng(seed, 5), count=cfg.lexical_gap_count)
    scenarios += contradiction.generate(_category_rng(seed, 1), count=cfg.contradiction_count)
    scenarios += multi_hop.generate(_category_rng(seed, 2), count=cfg.multi_hop_count)
    scenarios += recency_relevance.generate(_category_rng(seed, 3), count=cfg.recency_count)
    scenarios += needle.generate(
        _category_rng(seed, 4), count=cfg.needle_count, needle_depth=cfg.needle_depth
    )
    return Suite(
        suite=suite, dataset_version=DATASET_VERSION, seed=seed,
        scenarios=scenarios, config_name=cfg.name,
    )


# Categories, in dataset order (kept for reference / docs).
CATEGORY_ORDER = [
    LONGITUDINAL_RECALL, CONTRADICTION, MULTI_HOP, RECENCY_RELEVANCE, NEEDLE,
]


__all__ = ["build_suite", "DATASET_VERSION", "SuiteConfig", "PRESETS", "resolve_config"]

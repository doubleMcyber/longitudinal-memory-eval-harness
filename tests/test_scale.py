"""Suite scale/configurability tests (PRD §4, §8) — Wave 3."""

from __future__ import annotations

import pytest

from mem_eval.adapters import get_backend
from mem_eval.data.schema import suite_content_hash
from mem_eval.data.suite import PRESETS, SuiteConfig, build_suite
from mem_eval.runner.orchestrate import run_eval


def _sessions(s):
    return sum(len(x.sessions) for x in s.scenarios)


def test_default_is_standard_preset():
    s = build_suite("v1", seed=42)
    assert s.config_name == "standard"


def test_scales_change_size_monotonically():
    small = build_suite("v1", seed=42, scale="small")
    standard = build_suite("v1", seed=42, scale="standard")
    large = build_suite("v1", seed=42, scale="large")
    assert len(small.scenarios) < len(standard.scenarios) < len(large.scenarios)
    assert _sessions(small) < _sessions(standard) < _sessions(large)


def test_scale_is_deterministic_and_distinct():
    # same seed + scale -> identical content
    assert suite_content_hash(build_suite("v1", seed=42, scale="large")) == \
        suite_content_hash(build_suite("v1", seed=42, scale="large"))
    # different scale -> different content
    assert suite_content_hash(build_suite("v1", seed=42, scale="small")) != \
        suite_content_hash(build_suite("v1", seed=42, scale="large"))


def test_unknown_scale_rejected():
    with pytest.raises(ValueError):
        build_suite("v1", seed=42, scale="enormous")


def test_scale_recorded_in_scorecard_and_hash():
    s_small = build_suite("v1", seed=42, scale="small")
    s_std = build_suite("v1", seed=42, scale="standard")
    sc_small = run_eval(get_backend("naive_rag"), s_small, k=10)
    sc_std = run_eval(get_backend("naive_rag"), s_std, k=10)
    assert sc_small["dataset"]["scale"] == "small"
    assert sc_std["dataset"]["scale"] == "standard"
    # scale participates in dataset identity -> different determinism hashes
    assert sc_small["determinism_hash"] != sc_std["determinism_hash"]


def test_explicit_config_overrides_scale():
    cfg = SuiteConfig(name="tiny", longitudinal_count=1, contradiction_count=2,
                      multi_hop_count=1, recency_count=1, needle_count=1, needle_depth=5)
    s = build_suite("v1", seed=1, config=cfg)
    assert s.config_name == "tiny"
    assert len(s.scenarios) == 1 + 2 + 1 + 1 + 1


def test_presets_exist():
    assert set(PRESETS) == {"small", "standard", "large"}

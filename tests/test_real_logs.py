"""Real conversation-log path (PRD §9): importer + annotation sample run
end-to-end through the SAME metrics."""

from __future__ import annotations

import os

import json

import pytest

from mem_eval.adapters import BASELINES, REGISTRY, STUBS, get_backend
from mem_eval.adapters.base import MemoryBackend
from mem_eval.data.importers.transcript import (
    build_scenario_from_logs,
    load_annotation,
    propose_labels,
)
from mem_eval.data.schema import Suite, suite_content_hash
from mem_eval.runner.orchestrate import run_eval

_HERE = os.path.dirname(__file__)
_TRANSCRIPT = os.path.join(_HERE, "fixtures", "real_logs", "transcript.json")
_SIDECAR = os.path.join(_HERE, "fixtures", "real_logs", "annotations.json")


def _real_suite() -> Suite:
    scenario = build_scenario_from_logs(_TRANSCRIPT, _SIDECAR)
    return Suite(suite="real_sample", dataset_version="real-0", seed=0, scenarios=[scenario])


def test_importer_builds_scenario():
    scenario = build_scenario_from_logs(_TRANSCRIPT, _SIDECAR)
    assert len(scenario.sessions) == 3
    assert scenario.queries and scenario.queries[0].gold_support
    # the supporting ref resolves to a real turn provenance
    assert ("r-s0", "r-s0-t000") in scenario.turn_to_fact


def test_real_logs_run_end_to_end_through_same_metrics():
    suite = _real_suite()
    sc = run_eval(get_backend("naive_rag"), suite, k=10)
    overall = sc["metrics"]["overall"]
    # the planted backup-email turn must be recalled (primary metric for §9 sample)
    assert overall["recall_at_k"] > 0.0
    # the same 7-metric scorecard is produced for real logs as for synthetic suites
    for key in ("recall_at_k", "precision_at_k", "contradiction_resolution_accuracy",
                "staleness", "latency_ms", "cost_per_query_usd", "storage"):
        assert key in overall
    # determinism hash present and stable
    assert sc["determinism_hash"]


def test_importer_is_deterministic():
    assert suite_content_hash(_real_suite()) == suite_content_hash(_real_suite())


def test_label_proposal_is_flagged_low_confidence():
    proposal = propose_labels(_TRANSCRIPT)
    assert proposal["low_confidence"] is True
    assert proposal["queries"] == []


# --- Stub adapters (PRD §5, Stage 6) ---------------------------------------


def test_stub_registry_complete():
    assert set(STUBS) == {"letta", "curated_brain"}
    assert REGISTRY == {**BASELINES, **STUBS}


@pytest.mark.parametrize("name", list(STUBS))
def test_stubs_are_interface_complete_but_raise(name):
    """Stubs must instantiate, satisfy the contract Protocol, and raise
    NotImplementedError on every required operation until wired (PRD §5)."""
    from datetime import datetime

    b = get_backend(name)
    assert isinstance(b, MemoryBackend)
    from mem_eval.adapters.base import Session

    with pytest.raises(NotImplementedError):
        b.reset()
    with pytest.raises(NotImplementedError):
        b.ingest(Session("s", datetime(2025, 1, 1), []))
    with pytest.raises(NotImplementedError):
        b.query("p", 5, datetime(2025, 1, 1))
    with pytest.raises(NotImplementedError):
        b.stats()
    # consolidate inherits the contract's default no-op (PRD §4.2), not a raise
    assert b.consolidate().usd == 0.0


# --- YAML sidecar branch (PRD §9 names YAML; JSON is the v1 default) --------


def test_yaml_sidecar_matches_json(tmp_path):
    yaml = pytest.importorskip("yaml")
    with open(_SIDECAR) as fh:
        data = json.load(fh)
    yml = tmp_path / "annotations.yaml"
    yml.write_text(yaml.safe_dump(data))
    assert load_annotation(str(yml)) == data

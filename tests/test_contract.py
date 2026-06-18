"""Contract-rule tests (PRD §4.3): the frozen adapter contract is enforced
across EVERY shipped baseline, not just one (PRD §5: all three implement §4)."""

from __future__ import annotations

import dataclasses
import inspect
from datetime import datetime

import pytest

from mem_eval.adapters import (
    BASELINES,
    NaiveRAG,
    Session,
    Turn,
    get_backend,
)
from mem_eval.adapters.base import MemoryItem
from mem_eval.data.suite import build_suite

# Baselines that actually retain/return items (NoMemory is intentionally empty).
STATEFUL = ["long_context", "naive_rag"]


def _two_sessions():
    a = Session("A", datetime(2025, 1, 1), [Turn("A-t0", "user", "Alice's phone number is 555-1234.")])
    b = Session("B", datetime(2025, 1, 10), [Turn("B-t0", "user", "Bob's phone number is 555-9999.")])
    return a, b


def test_data_types_are_frozen():
    # Turn and Session are frozen dataclasses (PRD §4.1).
    assert dataclasses.is_dataclass(Turn) and Turn.__dataclass_params__.frozen
    assert dataclasses.is_dataclass(Session) and Session.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        Turn("x", "user", "y").text = "z"  # type: ignore[misc]


@pytest.mark.parametrize("name", list(BASELINES))
def test_time_discipline_no_leakage(name):
    """query(as_of=T) must only use sessions with timestamp <= T (PRD §4.3.1).
    Enforced for every baseline that ships."""
    a, b = _two_sessions()
    b_backend = get_backend(name)
    b_backend.reset()
    b_backend.ingest(a)
    b_backend.ingest(b)
    res = b_backend.query("phone number", 10, datetime(2025, 1, 5))  # before B
    sources = {it.source_session for it in res.items}
    assert "B" not in sources, f"{name}: future session leaked past as_of"
    if name in STATEFUL:
        assert "A" in sources, f"{name}: visible past session not retrievable"


@pytest.mark.parametrize("name", list(BASELINES))
def test_isolation_reset_clears_state(name):
    """reset() fully clears state; no scenario leaks into another (PRD §4.3.3)."""
    a, b = _two_sessions()
    backend = get_backend(name)
    backend.reset()
    backend.ingest(a)
    backend.reset()
    assert backend.stats().item_count == 0, f"{name}: reset did not clear item_count"
    # after reset, a fresh ingest must not surface pre-reset content
    backend.ingest(b)
    res = backend.query("phone number", 10, datetime(2025, 1, 20))
    assert "A" not in {it.source_session for it in res.items}, f"{name}: state leaked past reset"


@pytest.mark.parametrize("name", STATEFUL)
def test_provenance_present_and_stable(name):
    """Every returned item carries source_session + timestamp; ids stable (PRD §4.3.2, §4.3.4)."""
    a, b = _two_sessions()
    backend = get_backend(name)
    backend.reset()
    backend.ingest(a)
    backend.ingest(b)
    res1 = backend.query("phone number", 10, datetime(2025, 1, 20))
    res2 = backend.query("phone number", 10, datetime(2025, 1, 20))
    assert res1.items, f"{name}: expected retrieval"
    for it in res1.items:
        assert isinstance(it, MemoryItem)
        assert it.source_session
        assert isinstance(it.timestamp, datetime)
    # determinism: identical id order across identical queries
    assert [it.item_id for it in res1.items] == [it.item_id for it in res2.items]


def test_no_memory_is_truly_empty():
    """The floor returns nothing and never leaks (PRD §5)."""
    a, b = _two_sessions()
    nm = get_backend("no_memory")
    nm.reset()
    nm.ingest(a)
    nm.ingest(b)
    res = nm.query("phone number", 10, datetime(2025, 1, 20))
    assert res.items == []


@pytest.mark.parametrize("name", list(BASELINES))
def test_no_gold_access_in_query_signature(name):
    """A backend's query receives only prompt, k, as_of (PRD §4.3.6)."""
    sig = inspect.signature(get_backend(name).query)
    assert list(sig.parameters) == ["prompt", "k", "as_of"], name


def test_no_gold_leaks_into_prompts():
    """Behavioral §4.3.6 guard: synthetic query prompts must not contain the gold
    answer verbatim, so a backend cannot 'retrieve' the answer from the probe."""
    suite = build_suite("v1", seed=42)
    leaks = []
    for scenario in suite.scenarios:
        for q in scenario.queries:
            if q.gold_answer and q.gold_answer.lower() in q.prompt.lower():
                leaks.append(q.query_id)
    assert not leaks, f"gold answer leaked into prompt(s): {leaks}"


@pytest.mark.parametrize("name", list(BASELINES))
def test_consolidate_default_is_noop(name):
    backend = get_backend(name)
    backend.reset()
    usage = backend.consolidate()
    assert usage.usd == 0.0

"""Structured data schema for scenarios and ground truth (PRD §3, §8).

The defining property of this harness: **ground truth is constructed, not
inferred**. Gold labels live in this structured layer (``Fact`` / ``Query``),
never in rendered text. The surface (natural-language) layer is optional and
labels never depend on it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

from mem_eval.adapters.base import MemoryItem, Session

# Category tags (PRD §6) -----------------------------------------------------
LONGITUDINAL_RECALL = "longitudinal_recall"
CONTRADICTION = "contradiction"
MULTI_HOP = "multi_hop"
RECENCY_RELEVANCE = "recency_relevance"
NEEDLE = "needle"

CATEGORIES = (
    LONGITUDINAL_RECALL,
    CONTRADICTION,
    MULTI_HOP,
    RECENCY_RELEVANCE,
    NEEDLE,
)

# Query-intent tags for category 4 (PRD §6.4)
INTENT_RECENCY = "recency"
INTENT_RELEVANCE = "relevance"


@dataclass(frozen=True)
class Fact:
    """The atomic unit of ground truth: an ``(entity, attribute, value)``
    assertion planted on a specific turn at ``valid_from``."""

    fact_id: str
    entity: str
    attribute: str
    value: str
    valid_from: datetime
    source_session: str
    source_turn: str
    # explicit update chain for the same (entity, attribute):
    supersedes: str | None = None  # fact_id this one replaces
    superseded_by: str | None = None  # fact_id that replaces this one
    is_distractor: bool = False


@dataclass(frozen=True)
class Query:
    """A probe issued at ``as_of`` with constructed gold labels (PRD §3, §8)."""

    query_id: str
    category: str
    prompt: str
    as_of: datetime
    gold_answer: str
    gold_support: tuple[str, ...]  # fact_ids that SHOULD surface (the support set)
    gold_superseded: tuple[str, ...] = ()  # fact_ids that must NOT win (contradiction)
    intent: str | None = None  # INTENT_RECENCY | INTENT_RELEVANCE (category 4)
    k: int | None = None  # optional per-query override of retrieval depth
    answer_aliases: tuple[str, ...] = ()  # accepted surface variants of gold_answer


@dataclass
class Scenario:
    """An ordered timeline of sessions plus planted gold structure and queries."""

    scenario_id: str
    category: str
    sessions: list[Session]
    facts: list[Fact]
    queries: list[Query]
    # Grading map: a returned MemoryItem's (source_session, source_turn) provenance
    # resolves to a planted fact_id here. Backends never see this.
    turn_to_fact: dict[tuple[str, str], str] = field(default_factory=dict)

    def fact_by_id(self) -> dict[str, Fact]:
        return {f.fact_id: f for f in self.facts}

    def resolve_item(self, item: MemoryItem) -> str | None:
        """Map a returned item back to the fact it provenances from, via
        (source_session, source_turn). Returns None for unmappable items
        (e.g. distractor filler with no planted fact)."""
        if item.source_turn is None:
            return None
        return self.turn_to_fact.get((item.source_session, item.source_turn))


@dataclass
class Suite:
    """A bundle of scenarios with dataset identity (PRD §8 versioning)."""

    suite: str
    dataset_version: str
    seed: int
    scenarios: list[Scenario]
    config_name: str = "standard"  # scale preset name (PRD §4 scale)

    def scenarios_for(self, category: str) -> list[Scenario]:
        return [s for s in self.scenarios if s.category == category]

    def all_queries(self) -> list[tuple[Scenario, Query]]:
        return [(s, q) for s in self.scenarios for q in s.queries]


# ---------------------------------------------------------------------------
# Determinism: content hash over the structured layer only (labels, not renders).
# ---------------------------------------------------------------------------


def _fact_repr(f: Fact) -> dict:
    return {
        "fact_id": f.fact_id,
        "entity": f.entity,
        "attribute": f.attribute,
        "value": f.value,
        "valid_from": f.valid_from.isoformat(),
        "supersedes": f.supersedes,
        "superseded_by": f.superseded_by,
        "is_distractor": f.is_distractor,
        "source_session": f.source_session,
        "source_turn": f.source_turn,
    }


def _query_repr(q: Query) -> dict:
    return {
        "query_id": q.query_id,
        "category": q.category,
        "prompt": q.prompt,
        "as_of": q.as_of.isoformat(),
        "gold_answer": q.gold_answer,
        "gold_support": list(q.gold_support),
        "gold_superseded": list(q.gold_superseded),
        "intent": q.intent,
        "k": q.k,
        "answer_aliases": list(q.answer_aliases),
    }


def scenario_content_hash(scenario: Scenario) -> str:
    """Hash the *structured* content (facts + queries + session timeline). The
    rendered surface text is included via session structure so a render change
    that alters which turn carries a fact is caught, but labels are gold."""
    payload = {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "sessions": [
            {
                "session_id": s.session_id,
                "timestamp": s.timestamp.isoformat(),
                "turns": [
                    {"turn_id": t.turn_id, "role": t.role, "text": t.text}
                    for t in s.turns
                ],
            }
            for s in scenario.sessions
        ],
        "facts": [_fact_repr(f) for f in scenario.facts],
        "queries": [_query_repr(q) for q in scenario.queries],
        "turn_to_fact": {f"{k[0]}|{k[1]}": v for k, v in sorted(scenario.turn_to_fact.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def suite_content_hash(suite: Suite) -> str:
    parts = [f"{suite.suite}:{suite.dataset_version}:{suite.seed}"]
    parts += [scenario_content_hash(s) for s in suite.scenarios]
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def superseded_as_of(scenario: Scenario, as_of: datetime) -> set[str]:
    """Set of fact_ids that are superseded as of ``as_of``: a fact is superseded
    when a later fact for the same (entity, attribute) has valid_from <= as_of."""
    by_key: dict[tuple[str, str], list[Fact]] = {}
    for f in scenario.facts:
        if f.is_distractor:
            continue
        by_key.setdefault((f.entity, f.attribute), []).append(f)
    out: set[str] = set()
    for facts in by_key.values():
        visible = sorted(
            [f for f in facts if f.valid_from <= as_of], key=lambda f: f.valid_from
        )
        # all but the latest visible version are superseded
        for f in visible[:-1]:
            out.add(f.fact_id)
    return out


__all__ = [
    "Fact",
    "Query",
    "Scenario",
    "Suite",
    "CATEGORIES",
    "LONGITUDINAL_RECALL",
    "CONTRADICTION",
    "MULTI_HOP",
    "RECENCY_RELEVANCE",
    "NEEDLE",
    "INTENT_RECENCY",
    "INTENT_RELEVANCE",
    "scenario_content_hash",
    "suite_content_hash",
    "superseded_as_of",
]

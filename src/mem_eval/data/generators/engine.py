"""Shared planting engine for the structured ground-truth layer (PRD §8).

Generators build scenarios by planting ``Fact`` objects on turns and deriving
``Query`` objects with exact gold labels. This module supplies deterministic
vocabulary, id minting, and scenario assembly so each category generator stays
small and focused on its scenario shape. Labels never depend on rendered text.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from random import Random

from mem_eval.adapters.base import Session, Turn
from mem_eval.data.schema import Fact, Scenario

BASE_DATE = datetime(2025, 1, 1)


def session_timestamp(day_index: int) -> datetime:
    return BASE_DATE + timedelta(days=day_index)


# --- deterministic vocabulary ------------------------------------------------

ENTITIES = [
    "Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Peggy",
    "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xena",
    "Yusuf", "Zoe",
]

# (attribute phrase, value generator key)
ATTRIBUTES = [
    ("phone number", "phone"),
    ("email address", "email"),
    ("home city", "city"),
    ("job title", "title"),
    ("favorite color", "color"),
    ("car model", "car"),
    ("office building", "building"),
    ("preferred airline", "airline"),
]

_CITIES = ["Berlin", "Lagos", "Quito", "Osaka", "Cairo", "Lima", "Oslo", "Perth",
           "Accra", "Riga", "Hanoi", "Tunis", "Sofia", "Kyoto", "Bogota", "Tbilisi"]
_TITLES = ["analyst", "engineer", "designer", "auditor", "surgeon", "pilot",
           "botanist", "archivist", "machinist", "cartographer"]
_COLORS = ["crimson", "teal", "amber", "indigo", "olive", "scarlet", "magenta", "cyan"]
_CARS = ["sedan", "coupe", "roadster", "hatchback", "pickup", "minivan", "wagon"]
_BUILDINGS = ["Birchtower", "Coppergate", "Flintwall", "Glasshcollow", "Ironreach", "Larchspire"]
_AIRLINES = ["Skybridge", "Aerocrest", "Windjet", "Cloudline", "Stratus", "Zephyrair"]
_ORGS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Soylent",
         "Tyrell", "Cyberdyne", "Wonka", "Aperture"]


def _value(rng: Random, key: str) -> str:
    if key == "phone":
        return f"555-{rng.randint(1000, 9999)}"
    if key == "email":
        return f"user{rng.randint(100, 999)}@example.com"
    if key == "city":
        return rng.choice(_CITIES)
    if key == "title":
        return rng.choice(_TITLES)
    if key == "color":
        return rng.choice(_COLORS)
    if key == "car":
        return rng.choice(_CARS)
    if key == "building":
        return rng.choice(_BUILDINGS)
    if key == "airline":
        return rng.choice(_AIRLINES)
    raise KeyError(key)


class Vocab:
    """Draws non-colliding entities/attributes/values deterministically."""

    def __init__(self, rng: Random) -> None:
        self.rng = rng
        self._entities = list(ENTITIES)
        rng.shuffle(self._entities)
        self._ei = 0

    def entity(self) -> str:
        e = self._entities[self._ei % len(self._entities)]
        self._ei += 1
        return e

    def attribute(self) -> tuple[str, str]:
        return self.rng.choice(ATTRIBUTES)

    def value(self, key: str) -> str:
        return _value(self.rng, key)

    def org(self) -> str:
        return self.rng.choice(_ORGS)

    def city(self) -> str:
        return self.rng.choice(_CITIES)


# --- scenario assembly -------------------------------------------------------


class Planted:
    """Accumulates planted (session, turn, fact) tuples and assembles a Scenario,
    auto-building the (session_id, turn_id) -> fact_id grading map."""

    def __init__(self, scenario_id: str, category: str) -> None:
        self.scenario_id = scenario_id
        self.category = category
        self._sessions: dict[str, tuple[datetime, list[Turn]]] = {}
        self._order: list[str] = []
        self.facts: list[Fact] = []
        self.queries: list = []
        self._turn_to_fact: dict[tuple[str, str], str] = {}

    def session(self, day_index: int) -> str:
        sid = f"{self.scenario_id}-s{day_index:03d}"
        if sid not in self._sessions:
            self._sessions[sid] = (session_timestamp(day_index), [])
            self._order.append(sid)
        return sid

    def add_turn(self, sid: str, text: str, *, role: str = "user", fact: Fact | None = None) -> str:
        _, turns = self._sessions[sid]
        turn_id = f"{sid}-t{len(turns):03d}"
        turns.append(Turn(turn_id=turn_id, role=role, text=text))
        if fact is not None:
            # rebind fact provenance to the actual session/turn it landed on
            bound = Fact(
                fact_id=fact.fact_id,
                entity=fact.entity,
                attribute=fact.attribute,
                value=fact.value,
                valid_from=self._sessions[sid][0],
                source_session=sid,
                source_turn=turn_id,
                supersedes=fact.supersedes,
                superseded_by=fact.superseded_by,
                is_distractor=fact.is_distractor,
            )
            self.facts.append(bound)
            self._turn_to_fact[(sid, turn_id)] = bound.fact_id
        return turn_id

    def link_supersession(self, old_id: str, new_id: str) -> None:
        for i, f in enumerate(self.facts):
            if f.fact_id == old_id:
                self.facts[i] = _replace_fact(f, superseded_by=new_id)
            elif f.fact_id == new_id:
                self.facts[i] = _replace_fact(f, supersedes=old_id)

    def add_query(self, query) -> None:
        self.queries.append(query)

    def build(self) -> Scenario:
        sessions = [
            Session(session_id=sid, timestamp=self._sessions[sid][0], turns=self._sessions[sid][1])
            for sid in self._order
        ]
        return Scenario(
            scenario_id=self.scenario_id,
            category=self.category,
            sessions=sessions,
            facts=self.facts,
            queries=self.queries,
            turn_to_fact=dict(self._turn_to_fact),
        )


def _replace_fact(f: Fact, **changes) -> Fact:
    data = dict(
        fact_id=f.fact_id, entity=f.entity, attribute=f.attribute, value=f.value,
        valid_from=f.valid_from, source_session=f.source_session, source_turn=f.source_turn,
        supersedes=f.supersedes, superseded_by=f.superseded_by, is_distractor=f.is_distractor,
    )
    data.update(changes)
    return Fact(**data)


def new_fact(fact_id: str, entity: str, attribute: str, value: str, *, is_distractor: bool = False) -> Fact:
    # provenance is filled in by Planted.add_turn when the fact lands on a turn
    return Fact(
        fact_id=fact_id, entity=entity, attribute=attribute, value=value,
        valid_from=BASE_DATE, source_session="", source_turn="", is_distractor=is_distractor,
    )


__all__ = [
    "BASE_DATE", "session_timestamp", "ENTITIES", "ATTRIBUTES", "Vocab",
    "Planted", "new_fact",
]

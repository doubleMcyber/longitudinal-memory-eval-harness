"""Curation-stress scenarios (PRD §1 curation quality, §7.7 storage growth).

Memory that grows without bound degrades silently. This builds a timeline where a
single canonical fact is buried under a flood of EXACT-duplicate reminder noise.
A curating backend (one with a non-trivial `consolidate()`) compacts the redundant
copies and its storage footprint stays flat; an uncurated store grows linearly.
The growth-slope metric then separates them.

Kept out of the 5-category gate suite (so A2's "exactly 5 categories" holds); used
by the curation test and available for ad-hoc storage analysis. Gold is structured
and untouched by deduplication (the canonical fact is stated once).
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact, session_timestamp
from mem_eval.data.schema import LONGITUDINAL_RECALL, Query

_NOISE = "Reminder: sync the shared team calendar before standup."


def generate(rng: Random, count: int = 1, noise_sessions: int = 40) -> list:
    scenarios = []
    for i in range(count):
        voc = Vocab(Random(rng.randint(0, 2**31)))
        p = Planted(f"curation-{i}", LONGITUDINAL_RECALL)
        entity = voc.entity()
        attr, key = voc.attribute()
        value = voc.value(key)

        s0 = p.session(0)
        gold = new_fact(f"curation-{i}-gold", entity, attr, value)
        p.add_turn(s0, f"{entity}'s {attr} is {value}.", fact=gold)

        # a flood of identical reminder noise — redundant, carries no new fact
        for d in range(1, noise_sessions + 1):
            sd = p.session(d)
            nf = new_fact(f"curation-{i}-n{d}", "noise", "reminder", "calendar", is_distractor=True)
            p.add_turn(sd, _NOISE, fact=nf)

        p.add_query(
            Query(
                query_id=f"curation-{i}-q",
                category=LONGITUDINAL_RECALL,
                prompt=f"What is {entity}'s {attr}?",
                as_of=session_timestamp(noise_sessions),
                gold_answer=value,
                gold_support=(gold.fact_id,),
            )
        )
        scenarios.append(p.build())
    return scenarios


__all__ = ["generate"]

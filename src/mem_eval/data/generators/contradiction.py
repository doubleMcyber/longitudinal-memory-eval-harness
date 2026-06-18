"""Category 2 — Contradiction handling (PRD §6.2). The discrimination crux.

A fact is planted (the 'previous' value) then updated in a later session (the
'current' value), creating a superseded version. A query after the update must
return the CURRENT value and not the stale one.

Gold = current fact id + the set of superseded ids that must NOT win.
Primary metrics: contradiction-resolution accuracy, staleness.

Why NaiveRAG lands strictly between the NoMemory floor and a contradiction-aware
backend: it ranks by pure cosine with a recency tie-break and cannot tell a
*stale* fact about the target entity from *current* facts about other entities
with the same attribute. Two scenario shapes, designed so discrimination is
ROBUST across a wide band of k (not tuned to one k):

* crowded — many same-attribute 'current' distractors for OTHER entities are
  planted AFTER the superseded fact. They tie the superseded fact on cosine but
  are more recent, so the recency tie-break sinks the stale fact below the top-k
  cutoff for any k up to ~CROWDED_DISTRACTORS+1: NaiveRAG passes *by accident*
  (staleness 0 here) — not by any contradiction handling.
* sparse  — NO same-attribute distractors: the stale fact sits immediately
  behind the current one (rank 2) and is therefore returned for ANY k >= 2:
  NaiveRAG fails and exhibits staleness > 0. (Unrelated-attribute filler keeps
  the timeline realistic; it has zero cosine to the probe and never competes.)

A real contradiction-aware backend would pass BOTH by marking superseded facts.
Net: contradiction accuracy ~= 0.5 and staleness > 0 for NaiveRAG across
k in [2, CROWDED_DISTRACTORS+1]; the NoMemory floor is 0 on both.
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import (
    ATTRIBUTES,
    Planted,
    Vocab,
    new_fact,
    session_timestamp,
)
from mem_eval.data.schema import CONTRADICTION, Query

# Enough same-attribute distractors that the stale fact stays below the cutoff
# for any k up to CROWDED_DISTRACTORS + 1 (covers the default k=10 with margin).
CROWDED_DISTRACTORS = 20
SPARSE_FILLER = 6  # unrelated-attribute filler sessions (do not compete on cosine)


def _other_attribute(voc: Vocab, key: str):
    """Pick an attribute whose value-key differs from `key` so filler facts share
    no tokens with the probe (zero cosine -> never retrieved)."""
    for attr, k in ATTRIBUTES:
        if k != key:
            return attr, k
    return ATTRIBUTES[0]


def _scenario(rng: Random, idx: int, crowded: bool):
    voc = Vocab(Random(rng.randint(0, 2**31)))
    p = Planted(f"contra-{idx}", CONTRADICTION)
    entity = voc.entity()
    attr, key = voc.attribute()
    old_value = voc.value(key)
    new_value = voc.value(key)
    while new_value == old_value:
        new_value = voc.value(key)

    # day 0: the soon-to-be-superseded ('previous') value
    s0 = p.session(0)
    old = new_fact(f"contra-{idx}-old", entity, attr, old_value)
    p.add_turn(s0, f"{entity}'s previous {attr} was {old_value}.", fact=old)

    if crowded:
        # days 1..n: same-attribute 'current' distractors for OTHER entities
        n = CROWDED_DISTRACTORS
        for day in range(1, n + 1):
            de = voc.entity()
            dv = voc.value(key)
            sd = p.session(day)
            df = new_fact(f"contra-{idx}-d{day}", de, attr, dv, is_distractor=True)
            p.add_turn(sd, f"{de}'s current {attr} is {dv}.", fact=df)
        upd_day = n + 1
    else:
        # unrelated-attribute filler: realistic timeline, zero cosine to the probe
        fattr, fkey = _other_attribute(voc, key)
        for day in range(1, SPARSE_FILLER + 1):
            de = voc.entity()
            dv = voc.value(fkey)
            sd = p.session(day)
            df = new_fact(f"contra-{idx}-f{day}", de, fattr, dv, is_distractor=True)
            p.add_turn(sd, f"{de}'s {fattr} is {dv}.", fact=df)
        upd_day = SPARSE_FILLER + 1

    # update day: the 'current' value supersedes the old one
    su = p.session(upd_day)
    new = new_fact(f"contra-{idx}-new", entity, attr, new_value)
    p.add_turn(su, f"{entity}'s current {attr} is {new_value}.", fact=new)
    p.link_supersession(old.fact_id, new.fact_id)

    p.add_query(
        Query(
            query_id=f"contra-{idx}-q",
            category=CONTRADICTION,
            prompt=f"What is {entity}'s current {attr}?",
            as_of=session_timestamp(upd_day),
            gold_answer=new_value,
            gold_support=(new.fact_id,),
            gold_superseded=(old.fact_id,),
        )
    )
    return p.build()


def generate(rng: Random, count: int = 6) -> list:
    scenarios = []
    for i in range(count):
        crowded = (i % 2 == 0)  # alternate crowded / sparse -> accuracy in (0,1)
        scenarios.append(_scenario(rng, i, crowded))
    return scenarios


__all__ = ["generate", "CROWDED_DISTRACTORS", "SPARSE_FILLER"]

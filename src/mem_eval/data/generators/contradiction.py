"""Category 2 — Contradiction handling (PRD §6.2). The discrimination crux.

A fact is planted then updated one or more times in later sessions, creating a
chain of superseded versions. A query after the last update must return the
CURRENT value and not any stale one.

Gold = current fact id + the set of superseded ids that must NOT win.
Primary metrics: contradiction-resolution accuracy, staleness.

EMERGENT, not templated, AND not tuned to the reference. Two difficulty axes,
both RNG-driven and difficulty-stratified with fixed anchors that guarantee the
ordering for every seed while leaving the aggregate emergent:

1. Crowding / chain length — how many same-topic distractors and how long the
   update chain. Controls whether NaiveRAG (no supersession mechanism) accidentally
   sheds stale versions; its accuracy varies by seed.
2. Phrasing — `possessive` ("Alice's current phone number is X", which the
   reference's surface parser can consolidate) vs `coreference` ("Their current
   phone number is X", which drops the subject so the reference CANNOT bind the
   update to the chain). Coreference scenarios leave headroom ABOVE the reference:
   TemporalRAG cannot resolve them, so its contradiction accuracy is emergent and
   strictly < 1.0 — the benchmark is demonstrably NOT co-designed to let its own
   reference win.

Anchors (every seed): scenario 0 easy (naive passes by crowding, reference passes
by consolidation); scenario 1 hard-for-naive (long chain, no crowding — naive
fails, reference passes) guarantees reference > naive; scenario 2 hard-for-
reference (coreference, no crowding — both fail) guarantees reference < 1.0.
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

MAX_UPDATES = 3          # chain length 1..MAX_UPDATES (superseded versions)
# Same-topic 'current' distractors for other entities. The easy anchor uses the
# max so the stale fact is pushed below the top-k cutoff across the tested k-band
# (>= 20), guaranteeing NaiveRAG passes at least one scenario for every seed.
MAX_DISTRACTORS = 24
COREF_RATE = 0.4         # fraction of RNG scenarios that use coreference phrasing

_OLD_TEMPLATES = [
    "{e}'s previous {a} was {v}.",
    "Earlier, {e}'s {a} was {v}.",
    "{e}'s {a} was {v} at the time.",
]
_MID_TEMPLATES = [
    "{e}'s {a} was changed to {v}.",
    "{e}'s {a} was then {v}.",
    "After that, {e}'s {a} was {v}.",
]


def _distinct_values(voc: Vocab, key: str, n: int) -> list[str]:
    vals: list[str] = []
    guard = 0
    while len(vals) < n and guard < 200:
        v = voc.value(key)
        if v not in vals:
            vals.append(v)
        guard += 1
    while len(vals) < n:
        vals.append(f"{vals[-1]}-{len(vals)}")
    return vals


def _difficulty(rng: Random, idx: int) -> tuple[int, int, str]:
    """(chain length, distractor count, phrasing). Three anchors pin the gradient
    for EVERY seed; the rest are RNG-drawn so the aggregate is emergent."""
    if idx == 0:
        return 1, MAX_DISTRACTORS, "possessive"        # easy: naive passes by crowding
    if idx == 1:
        return MAX_UPDATES, 0, "possessive"            # hard-for-naive: reference > naive
    if idx == 2:
        return 1, 0, "coreference"                     # hard-for-reference: reference < 1.0
    phrasing = "coreference" if rng.random() < COREF_RATE else "possessive"
    return rng.randint(1, MAX_UPDATES), rng.randint(0, MAX_DISTRACTORS), phrasing


def _scenario(rng: Random, idx: int):
    voc = Vocab(Random(rng.randint(0, 2**31)))
    p = Planted(f"contra-{idx}", CONTRADICTION)
    entity = voc.entity()
    attr, key = voc.attribute()

    n_updates, n_distractors, phrasing = _difficulty(rng, idx)
    versions = _distinct_values(voc, key, n_updates + 1)

    day = 0
    chain_ids: list[str] = []

    s0 = p.session(day)
    old0 = new_fact(f"contra-{idx}-v0", entity, attr, versions[0])
    p.add_turn(s0, rng.choice(_OLD_TEMPLATES).format(e=entity, a=attr, v=versions[0]), fact=old0)
    chain_ids.append(old0.fact_id)
    day += 1

    for d in range(day, day + n_distractors):
        de = voc.entity()
        dv = voc.value(key)
        sd = p.session(d)
        df = new_fact(f"contra-{idx}-d{d}", de, attr, dv, is_distractor=True)
        p.add_turn(sd, f"{de}'s current {attr} is {dv}.", fact=df)
    day += n_distractors

    for i in range(1, n_updates):
        sd = p.session(day)
        mid = new_fact(f"contra-{idx}-v{i}", entity, attr, versions[i])
        p.add_turn(sd, rng.choice(_MID_TEMPLATES).format(e=entity, a=attr, v=versions[i]), fact=mid)
        chain_ids.append(mid.fact_id)
        day += 1

    # the current value (most recent) supersedes everything before it.
    # In coreference phrasing the subject is a pronoun, so a surface parser cannot
    # bind this update to the entity's chain (the reference's consolidation fails).
    sd = p.session(day)
    cur = new_fact(f"contra-{idx}-cur", entity, attr, versions[n_updates])
    if phrasing == "coreference":
        cur_text = f"Their current {attr} is {versions[n_updates]}."
    else:
        cur_text = f"{entity}'s current {attr} is {versions[n_updates]}."
    p.add_turn(sd, cur_text, fact=cur)
    for old_id in chain_ids:
        p.link_supersession(old_id, cur.fact_id)

    p.add_query(
        Query(
            query_id=f"contra-{idx}-q",
            category=CONTRADICTION,
            prompt=f"What is {entity}'s current {attr}?",
            as_of=session_timestamp(day),
            gold_answer=versions[n_updates],
            gold_support=(cur.fact_id,),
            gold_superseded=tuple(chain_ids),
        )
    )
    return p.build()


def generate(rng: Random, count: int = 10) -> list:
    return [_scenario(rng, i) for i in range(count)]


__all__ = ["generate", "MAX_UPDATES", "MAX_DISTRACTORS", "COREF_RATE"]

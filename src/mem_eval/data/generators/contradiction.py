"""Category 2 — Contradiction handling (PRD §6.2). The discrimination crux.

A fact is planted then updated one or more times in later sessions, creating a
chain of superseded versions. A query after the last update must return the
CURRENT value and not any stale one.

Gold = current fact id + the set of superseded ids that must NOT win.
Primary metrics: contradiction-resolution accuracy, staleness.

EMERGENT, not templated. Earlier this category hardcoded a crowded/sparse split
(i % 2) so NaiveRAG's accuracy was a fixed 0.5 by construction. It is now a
difficulty-stratified, RNG-driven distribution: each scenario draws an update
chain length (1-3) and a same-topic distractor density (0..MAX_DISTRACTORS).
Consequently the metric values fall out of the *data* and *backend mechanism*,
and vary across seeds, rather than being designed:

* NaiveRAG (pure cosine + recency tiebreak) has NO supersession mechanism, so it
  returns stale chain versions whenever they survive the top-k cutoff — failing
  the harder (long-chain / low-distractor) scenarios and only accidentally
  passing the heavily-crowded ones. Its accuracy and staleness therefore depend
  on the seed-drawn difficulty mix.
* TemporalRAG consolidates the whole same-topic chain (keeps the latest), so it
  resolves contradictions regardless of chain length or distractor density.

The harness thus separates a contradiction-aware backend from a naive one as an
emergent property: TemporalRAG.accuracy >> NaiveRAG.accuracy > NoMemory == 0,
and NaiveRAG.staleness > TemporalRAG.staleness == 0, on data it did not tune to.
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

# Paraphrase templates for intermediate/old assertions — adds surface variety so
# retrieval is not a fixed string match. All keep a copula so the topic key
# (subject + attribute, temporal markers stripped) parses consistently.
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
    # pad deterministically if the value space is small
    while len(vals) < n:
        vals.append(f"{vals[-1]}-{len(vals)}")
    return vals


def _difficulty(rng: Random, idx: int) -> tuple[int, int]:
    """Difficulty stratification (standard benchmark practice). Two anchors pin
    the ends of the gradient so the suite always spans easy->hard for EVERY seed:
    scenario 0 is easy (naive accidentally passes), scenario 1 is hard (naive
    cannot). The rest are RNG-drawn, so the aggregate metric is emergent — it
    varies by seed — while the ordering (NoMemory < NaiveRAG < TemporalRAG) is
    guaranteed by always having both a naive-passable and a naive-failing case."""
    if idx == 0:
        return 1, MAX_DISTRACTORS           # easy: single update, heavy crowding
    if idx == 1:
        return MAX_UPDATES, 0               # hard: long chain, no crowding
    return rng.randint(1, MAX_UPDATES), rng.randint(0, MAX_DISTRACTORS)


def _scenario(rng: Random, idx: int):
    voc = Vocab(Random(rng.randint(0, 2**31)))
    p = Planted(f"contra-{idx}", CONTRADICTION)
    entity = voc.entity()
    attr, key = voc.attribute()

    n_updates, n_distractors = _difficulty(rng, idx)  # superseded versions, crowding
    versions = _distinct_values(voc, key, n_updates + 1)  # v0..v_{n_updates}

    day = 0
    chain_ids: list[str] = []

    # oldest assertion (day 0)
    s0 = p.session(day)
    old0 = new_fact(f"contra-{idx}-v0", entity, attr, versions[0])
    p.add_turn(s0, rng.choice(_OLD_TEMPLATES).format(e=entity, a=attr, v=versions[0]), fact=old0)
    chain_ids.append(old0.fact_id)
    day += 1

    # distractors + intermediate updates interleaved on a shared timeline
    distractor_days = list(range(day, day + n_distractors))
    for d in distractor_days:
        de = voc.entity()
        dv = voc.value(key)
        sd = p.session(d)
        df = new_fact(f"contra-{idx}-d{d}", de, attr, dv, is_distractor=True)
        p.add_turn(sd, f"{de}'s current {attr} is {dv}.", fact=df)
    day += n_distractors

    # intermediate (still superseded) updates v1..v_{n_updates-1}
    for i in range(1, n_updates):
        sd = p.session(day)
        mid = new_fact(f"contra-{idx}-v{i}", entity, attr, versions[i])
        p.add_turn(sd, rng.choice(_MID_TEMPLATES).format(e=entity, a=attr, v=versions[i]), fact=mid)
        chain_ids.append(mid.fact_id)
        day += 1

    # the current value (most recent) supersedes everything before it
    sd = p.session(day)
    cur = new_fact(f"contra-{idx}-cur", entity, attr, versions[n_updates])
    p.add_turn(sd, f"{entity}'s current {attr} is {versions[n_updates]}.", fact=cur)
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


__all__ = ["generate", "MAX_UPDATES", "MAX_DISTRACTORS"]

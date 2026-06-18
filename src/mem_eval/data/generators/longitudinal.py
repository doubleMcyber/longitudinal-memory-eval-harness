"""Category 1 — Longitudinal recall (PRD §6.1).

A fact is planted in an early session and queried many sessions later with no
reinforcement. Gold = the originating fact's id. Primary metric: recall@k.
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact
from mem_eval.data.schema import LONGITUDINAL_RECALL, Query


def generate(rng: Random, count: int = 3, num_sessions: int = 12) -> list:
    scenarios = []
    for i in range(count):
        voc = Vocab(Random(rng.randint(0, 2**31)))
        p = Planted(f"long-{i}", LONGITUDINAL_RECALL)
        entity = voc.entity()
        attr, key = voc.attribute()
        value = voc.value(key)
        # plant the target in the earliest session (day 0)
        s0 = p.session(0)
        tgt = new_fact(f"long-{i}-target", entity, attr, value)
        p.add_turn(s0, f"{entity}'s {attr} is {value}.", fact=tgt)

        # fill later sessions with unrelated distractors (no reinforcement)
        for day in range(1, num_sessions):
            de = voc.entity()
            da, dk = voc.attribute()
            dv = voc.value(dk)
            sd = p.session(day)
            df = new_fact(f"long-{i}-d{day}", de, da, dv, is_distractor=True)
            p.add_turn(sd, f"{de}'s {da} is {dv}.", fact=df)

        last_day = num_sessions - 1
        from mem_eval.data.generators.engine import session_timestamp

        p.add_query(
            Query(
                query_id=f"long-{i}-q",
                category=LONGITUDINAL_RECALL,
                prompt=f"What is {entity}'s {attr}?",
                as_of=session_timestamp(last_day),
                gold_answer=value,
                gold_support=(tgt.fact_id,),
            )
        )
        scenarios.append(p.build())
    return scenarios


__all__ = ["generate"]

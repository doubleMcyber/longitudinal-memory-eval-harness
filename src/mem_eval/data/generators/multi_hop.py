"""Category 3 — Multi-hop retrieval (PRD §6.3).

The answer requires combining >=2 facts planted in DIFFERENT sessions. Gold =
the full supporting set {A, B}. Primary: recall of the complete support set and
answer accuracy. (A retriever can surface both hops yet still fail to synthesize
the combined answer — exactly the gap this category exposes.)
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact, session_timestamp
from mem_eval.data.schema import MULTI_HOP, Query


def generate(rng: Random, count: int = 3) -> list:
    scenarios = []
    for i in range(count):
        voc = Vocab(Random(rng.randint(0, 2**31)))
        p = Planted(f"hop-{i}", MULTI_HOP)
        person = voc.entity()
        org = voc.org()
        city = voc.city()

        # hop A in an early session
        sA = p.session(1)
        fa = new_fact(f"hop-{i}-A", person, "employer", org)
        p.add_turn(sA, f"{person} works at {org}.", fact=fa)

        # filler distractor sessions between the hops
        for day in (0, 2, 3, 4):
            de = voc.entity()
            da, dk = voc.attribute()
            dv = voc.value(dk)
            sd = p.session(day)
            df = new_fact(f"hop-{i}-d{day}", de, da, dv, is_distractor=True)
            p.add_turn(sd, f"{de}'s {da} is {dv}.", fact=df)

        # hop B in a later session
        sB = p.session(5)
        fb = new_fact(f"hop-{i}-B", org, "headquarters", city)
        p.add_turn(sB, f"{org} is headquartered in {city}.", fact=fb)

        p.add_query(
            Query(
                query_id=f"hop-{i}-q",
                category=MULTI_HOP,
                prompt=f"Where is {org}, the company {person} works for, headquartered?",
                as_of=session_timestamp(5),
                gold_answer=city,
                gold_support=(fa.fact_id, fb.fact_id),
            )
        )
        scenarios.append(p.build())
    return scenarios


__all__ = ["generate"]

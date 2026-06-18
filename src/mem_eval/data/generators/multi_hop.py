"""Category 3 — Multi-hop retrieval (PRD §6.3).

The answer requires combining facts planted in DIFFERENT sessions. Gold = the full
supporting set; the answer is the TERMINAL hop's value. Primary: recall of the
complete support set and answer accuracy.

Chain length is RNG-drawn (2 or 3 hops) with RNG-placed session gaps, so chains
vary in length/difficulty rather than being one fixed shape. The category's real
discrimination is ASSEMBLY, not retrieval: the in-box lexical/curating backends
surface all the hops (recall ~1.0) but cannot combine them into the terminal
answer (answer_accuracy ~0.0) — leaving large, honest headroom for a backend that
can actually chain reasoning across sessions.
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact, session_timestamp
from mem_eval.data.schema import MULTI_HOP, Query

_COUNTRIES = ["Germany", "Nigeria", "Ecuador", "Japan", "Egypt", "Peru", "Norway",
              "Australia", "Ghana", "Latvia", "Vietnam", "Tunisia", "Bulgaria"]


def _scenario(rng: Random, idx: int):
    voc = Vocab(Random(rng.randint(0, 2**31)))
    p = Planted(f"hop-{idx}", MULTI_HOP)
    person = voc.entity()
    org = voc.org()
    city = voc.city()
    country = _COUNTRIES[rng.randrange(len(_COUNTRIES))]
    hops = rng.choice([2, 3])

    # spread the hops across non-adjacent sessions, with filler distractors between
    days = sorted(rng.sample(range(0, 8), hops))
    support: list[str] = []

    sA = p.session(days[0])
    fa = new_fact(f"hop-{idx}-A", person, "employer", org)
    p.add_turn(sA, f"{person} works at {org}.", fact=fa)
    support.append(fa.fact_id)

    sB = p.session(days[1])
    fb = new_fact(f"hop-{idx}-B", org, "headquarters", city)
    p.add_turn(sB, f"{org} is headquartered in {city}.", fact=fb)
    support.append(fb.fact_id)

    if hops == 3:
        sC = p.session(days[2])
        fc = new_fact(f"hop-{idx}-C", city, "country", country)
        p.add_turn(sC, f"{city} is located in {country}.", fact=fc)
        support.append(fc.fact_id)
        answer = country
        prompt = f"In which country is the headquarters of {org}, where {person} works, located?"
    else:
        answer = city
        prompt = f"Where is {org}, the company {person} works for, headquartered?"

    # filler distractor sessions on the days not used by hops
    for day in range(0, max(days) + 2):
        if day in days:
            continue
        de = voc.entity()
        da, dk = voc.attribute()
        dv = voc.value(dk)
        sd = p.session(day)
        df = new_fact(f"hop-{idx}-d{day}", de, da, dv, is_distractor=True)
        p.add_turn(sd, f"{de}'s {da} is {dv}.", fact=df)

    p.add_query(
        Query(
            query_id=f"hop-{idx}-q",
            category=MULTI_HOP,
            prompt=prompt,
            as_of=session_timestamp(max(days) + 1),
            gold_answer=answer,
            gold_support=tuple(support),  # terminal hop is last (answer == its value)
        )
    )
    return p.build()


def generate(rng: Random, count: int = 3) -> list:
    return [_scenario(rng, i) for i in range(count)]


__all__ = ["generate"]

"""Category 4 — Recency-vs-relevance tradeoff (PRD §6.4).

A highly relevant-but-old item competes with a recent-but-marginal one. Correct
behavior depends on explicit query INTENT: "what's my current X?" => recency;
"what did I say about X back then?" => relevance. Gold = the intent-appropriate
item. Primary: precision@k, ranking quality.

NaiveRAG has no intent awareness — it always returns the most cosine-similar
item (the detailed old one), so it satisfies relevance-intent queries but FAILS
recency-intent queries. Per-query k=1 makes that ranking failure crisp.
"""

from __future__ import annotations

from random import Random

from mem_eval.data.generators.engine import Planted, Vocab, new_fact, session_timestamp
from mem_eval.data.schema import (
    INTENT_RECENCY,
    INTENT_RELEVANCE,
    RECENCY_RELEVANCE,
    Query,
)

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def generate(rng: Random, count: int = 2) -> list:
    scenarios = []
    for i in range(count):
        voc = Vocab(Random(rng.randint(0, 2**31)))
        p = Planted(f"recrel-{i}", RECENCY_RELEVANCE)
        person = voc.entity()
        old_when = _MONTHS[rng.randrange(len(_MONTHS))]
        new_when = _MONTHS[rng.randrange(len(_MONTHS))]
        while new_when == old_when:
            new_when = _MONTHS[rng.randrange(len(_MONTHS))]

        # old but highly relevant (detailed): contains "project deadline"
        s_old = p.session(1)
        f_old = new_fact(f"recrel-{i}-old", person, "project_deadline_detail", old_when)
        p.add_turn(s_old, f"{person}'s main project deadline is {old_when}.", fact=f_old)

        # filler sessions in between
        for day in (2, 3, 4, 5, 6, 7, 8, 9):
            de = voc.entity()
            da, dk = voc.attribute()
            dv = voc.value(dk)
            sd = p.session(day)
            df = new_fact(f"recrel-{i}-d{day}", de, da, dv, is_distractor=True)
            p.add_turn(sd, f"{de}'s {da} is {dv}.", fact=df)

        # recent but marginal mention
        s_new = p.session(10)
        f_new = new_fact(f"recrel-{i}-new", person, "project_deadline_recent", new_when)
        p.add_turn(s_new, f"{person} briefly noted the project changed to {new_when}.", fact=f_new)

        as_of = session_timestamp(10)
        # recency intent -> gold is the recent item
        p.add_query(
            Query(
                query_id=f"recrel-{i}-recency",
                category=RECENCY_RELEVANCE,
                prompt=f"What is {person}'s current project deadline?",
                as_of=as_of,
                gold_answer=new_when,
                gold_support=(f_new.fact_id,),
                intent=INTENT_RECENCY,
                k=1,
            )
        )
        # relevance intent -> gold is the detailed old item
        p.add_query(
            Query(
                query_id=f"recrel-{i}-relevance",
                category=RECENCY_RELEVANCE,
                prompt=f"What did {person} originally say the project deadline was?",
                as_of=as_of,
                gold_answer=old_when,
                gold_support=(f_old.fact_id,),
                intent=INTENT_RELEVANCE,
                k=1,
            )
        )
        scenarios.append(p.build())
    return scenarios


__all__ = ["generate"]
